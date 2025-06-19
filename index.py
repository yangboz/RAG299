from rag import Rag

rag = Rag('deepseek-r1:7b', 'pkb.txt')
msg = rag.chat('请介绍下刘芳')
print(msg)


