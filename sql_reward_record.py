import json
import time
import re
import psycopg2
import psycopg2.errors 
from typing import Any
import hashlib
import traceback
import os
from datetime import datetime

# ==============================================================================
# 0.  全局日志配置
# ==============================================================================
EXECUTION_LOG_FILE = "/data/shipei/downstream_rl/logs/sql_execution_details.jsonl"

def log_execution_detail(info_dict):
    try:
        os.makedirs(os.path.dirname(EXECUTION_LOG_FILE), exist_ok=True)
        info_dict['log_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(EXECUTION_LOG_FILE, "a", encoding='utf-8') as f:
            f.write(json.dumps(info_dict, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Write Log Error: {e}")

# ==============================================================================
# 1.  全局数据库配置
# ==============================================================================
UNIFIED_DB_CONFIG = {
    "user": "shipei",   
    "password": "021105",     
    "host": "localhost",
    "port": "5432"
}

DB_CONFIGS = {
    "solar_panel":            (UNIFIED_DB_CONFIG, "solar_panel"),
    "polar_equipment":        (UNIFIED_DB_CONFIG, "polar_equipment"),
    "robot_fault_prediction": (UNIFIED_DB_CONFIG, "robot_fault_prediction"),
    "unknown":                (UNIFIED_DB_CONFIG, "tpch_01g"), 
}

CONN_POOL = {}

# ==============================================================================
# 2.  核心工具函数
# ==============================================================================
def get_db_cursor(db_key):
    target_key = db_key if db_key in DB_CONFIGS else "unknown"
    instance_config, real_db_name = DB_CONFIGS[target_key]
    
    conn = CONN_POOL.get(target_key)
    # 检查连接是否存活
    if conn:
        try:
            if conn.closed != 0: raise Exception("Closed")
        except:
            if target_key in CONN_POOL: del CONN_POOL[target_key]
            conn = None
            
    if not conn:
        try:
            connect_args = instance_config.copy()
            connect_args["dbname"] = real_db_name
            conn = psycopg2.connect(**connect_args)
            conn.autocommit = False 
            CONN_POOL[target_key] = conn
        except Exception as e:
            return None, None, f"Connection Failed ({real_db_name}): {str(e)}"
            
    try:
        return conn, conn.cursor(), None
    except Exception as e:
        return None, None, f"Cursor Error: {str(e)}"

def apply_simple_patch(original_text, diff_text):
    if not original_text or not diff_text: return None
    try:
        original_lines = original_text.replace('\r\n', '\n').splitlines()
        diff_lines = diff_text.replace('\r\n', '\n').splitlines()
        result_lines = []
        src_idx = 0 
        i = 0
        while i < len(diff_lines):
            line = diff_lines[i]
            if line.startswith('---') or line.startswith('+++') or line.startswith('@@'):
                i += 1; continue
            if line.startswith(' '):
                if src_idx < len(original_lines):
                    result_lines.append(original_lines[src_idx]); src_idx += 1
            elif line.startswith('-'):
                src_idx += 1
            elif line.startswith('+'):
                result_lines.append(line[1:]) 
            i += 1
        while src_idx < len(original_lines):
            result_lines.append(original_lines[src_idx]); src_idx += 1
        return '\n'.join(result_lines).strip()
    except:
        return None

def extract_patch_from_response(response_str):
    if not response_str: return None
    # 优先匹配 Verified Patch 区块
    pattern = r"### Verified Patch:.*?```(?:diff|sql)?\s*(.*?)```"
    match = re.search(pattern, response_str, re.DOTALL | re.IGNORECASE)
    if match: return match.group(1).strip()
    
    # 兜底：找第一个代码块
    candidates = re.findall(r"```(?:diff|sql)?\s*(.*?)```", response_str, re.DOTALL)
    if candidates: return candidates[0].strip()
    return None

def execute_sql_bounded(cursor, sql, time_limit_ms):
    if not sql: return None, 0, None, "Empty SQL"
    # Audit 任务主要看正确性，给一个宽松的超时时间 (比如基准时间 + 2秒)
    # 或者直接给个固定值 5000ms，防止死循环即可
    timeout_val = 2000 
    
    start = time.time()
    try:
        cursor.execute(f"SET statement_timeout = {timeout_val};")
        cursor.execute(sql)
        rows = []
        if cursor.description:
            rows = cursor.fetchall()
        duration = (time.time() - start) * 1000 
        
        # 结果序列化用于 Hash 对比
        try: sorted_rows = str(sorted(rows, key=lambda x: str(x)))
        except: sorted_rows = str(rows)
            
        return duration, len(rows), sorted_rows, None
    except psycopg2.errors.QueryCanceled:
        return None, 0, None, "TIMEOUT_EXCEEDED"
    except Exception as e:
        return None, 0, None, str(e)

def normalize_sql(s):
    """
    标准化 SQL：去掉多余空格、转小写，用于检测是否回退
    """
    if not s: return ""
    return " ".join(s.strip().split()).lower()

# ==============================================================================
# 3. Reward Function (Audit & Fix 专用版)
# ==============================================================================
def sql_optimize(data_source, solution_str, ground_truth, extra_info=None):
    log_info = {
        "status": "init",
        "reward": 0.0,
        "error_msg": None,
        "generated_sql": None,
        "db_name": ground_truth.get('db', "unknown") if ground_truth else "unknown",
        "base_sql": ground_truth.get('base_sql', "") if ground_truth else ""
    }

    # 1. 解析 Ground Truth
    if not ground_truth:
        log_info["status"] = "missing_gt"
        log_execution_detail(log_info)
        return 0.0
        
    base_sql = ground_truth.get('base_sql', "")
    db_key = ground_truth.get('db', "unknown")
    gt_res_hash = ground_truth.get('base_result_hash')
    # 对于 Audit 任务，基准时间仅作参考，不参与核心判分
    gt_time = ground_truth.get('base_exec_time', 1000) 

    if gt_res_hash is None:
        log_info["status"] = "missing_gt_hash"
        log_execution_detail(log_info)
        return 0.0

    # 2. 解析 Patch
    patch_content = extract_patch_from_response(solution_str)
    if not patch_content:
        log_info["status"] = "patch_extract_failed"
        log_info["reward"] = -1.0
        log_execution_detail(log_info)
        return -1.0 

    pred_sql = apply_simple_patch(base_sql, patch_content)
    log_info["generated_sql"] = pred_sql

    if not pred_sql:
        log_info["status"] = "patch_apply_failed"
        log_info["reward"] = -1.0
        log_execution_detail(log_info)
        return -1.0 

    # 3. 连接数据库
    conn, cursor, conn_err = get_db_cursor(db_key)
    if conn_err:
        log_info["status"] = "db_connect_failed"
        log_info["error_msg"] = conn_err
        log_execution_detail(log_info)
        return 0.0 

    reward = 0.0
    try:
        conn.rollback()
        
        # 执行 SQL
        pred_time, _, pred_res, pred_err = execute_sql_bounded(cursor, pred_sql, gt_time)
        
        log_info["execution_time"] = pred_time
        log_info["error_msg"] = pred_err
        
        # --- 判分逻辑 (Audit & Fix Mode) ---

        if pred_err == "TIMEOUT_EXCEEDED":
            log_info["status"] = "timeout"
            reward = 0.1 

        elif pred_err:
            log_info["status"] = "sql_exec_error"
            reward = -1.0 # 语法错误

        else:
            # 跑通了，检查结果
            curr_hash = hashlib.md5(str(pred_res).encode()).hexdigest()
            
            if curr_hash == gt_res_hash:
                # 结果正确！现在检查是否回退
                is_reverted = normalize_sql(pred_sql) == normalize_sql(base_sql)
                
                if is_reverted:
                    # 直接回退到 Base SQL
                    log_info["status"] = "success_revert_bad"
                    reward = 0.1  # 惩罚分，告诉模型这不对
                else:
                    #  成功：结果正确，且保留了策略（代码不同于 Base）
                    log_info["status"] = "success_fixed_strategy"
                    reward = 1.0  # 满分
            else:
                # 结果不对
                log_info["status"] = "success_mismatch"
                reward = -0.5

    except Exception as e:
        error_content = f"{str(e)}\n{traceback.format_exc()}"
        log_info["status"] = "code_crash"
        log_info["error_msg"] = error_content
        reward = 0.0
    
    finally:
        if conn: 
            try: conn.rollback()
            except: pass

    log_info["reward"] = float(reward)
    log_execution_detail(log_info)
    
    return float(reward)
