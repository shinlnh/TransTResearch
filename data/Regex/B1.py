import re

text = "abc123xyz45"
result = re.findall(r"\d+", text)

print(result)



"""
Match tất cả các chuỗi số trong văn bản đã cho : 

abc123xyz45
"""