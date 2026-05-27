"""
Reuse the proven fix_json_format from autogen_base.py.
"""
import re

def fix_json_format(json_str, repair_attempt=1):
    json_pattern = r'(\{(?s).*?\}|(?s)\[(?s).*?\])'
    matches = re.findall(json_pattern, json_str, re.DOTALL)
    if matches:
        json_str = max(matches, key=len)
    else:
        json_str = re.sub(r'[^\{\}\[\],:"\'\\/\w\s.\-=+]', '', json_str)
    json_str = re.sub(r'//.*?(?=\n|$)', '', json_str, flags=re.MULTILINE)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    json_str = re.sub(r'(?<=[\'\"}\]]),\s*(?=[}\]])', '', json_str)
    json_str = re.sub(r'(?<=[\'\"}\]\d]),\s*(?=\])', '', json_str)
    if repair_attempt >= 1:
        json_str = re.sub(r'(?<="[^"\\]*)"(?=[^"\\]*")', '\\"', json_str)
        json_str = re.sub(r'(?<!\\)\\(?!["\\/bfnrt])', r'\\\\', json_str)
    if repair_attempt >= 2:
        json_str = re.sub(r'""([^"]+)""(?=:)', r'"\1"', json_str)
        json_str = re.sub(r'(?<=[\{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:',
                          r'"\1":', json_str)
        json_str = re.sub(r'[　\t]', ' ', json_str).strip()
    return json_str
