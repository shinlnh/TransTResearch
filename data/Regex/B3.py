import re 

text = "LiangYuHuiAI2025"

result = re.findall(r"[A-Z][A-Z]+[0-9]+", text)

print(result)


"""
Match chuỗi AI2025 trong văn bản đã cho :
LiangYuHuiAI2025
"""
