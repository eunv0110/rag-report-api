from langfuse import Langfuse
from dotenv import load_dotenv
load_dotenv()

lf = Langfuse()

# Langfuse 객체가 가진 메서드 목록
[m for m in dir(lf) if not m.startswith("_")]
