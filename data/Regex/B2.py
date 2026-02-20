import re 

text = "HelloWorld"

result = re.findall(r"[a-z]", text)

print(result)

"""
Match tất cả các chữ cái thường trong văn bản đã cho :
HelloWorld
"""