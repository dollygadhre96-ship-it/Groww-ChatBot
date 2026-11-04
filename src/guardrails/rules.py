"""Deterministic input/output guardrails for the factual FAQ demo."""
from dataclasses import dataclass
import re

PII_PATTERNS = [
 ("PAN", re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.I)),
 ("Aadhaar", re.compile(r"\b(?:\d[ -]?){12}\b")),
 ("OTP", re.compile(r"\b(?:otp|one[- ]time password)\s*(?:is|:)?\s*\d{4,8}\b", re.I)),
 ("phone", re.compile(r"(?<!\d)(?:\+?91[ -]?)?[6-9]\d{9}(?!\d)")),
 ("email", re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")),
 ("account number", re.compile(r"\b(?:account|a/c)\s*(?:number|no\.?|#)?\s*[:=-]?\s*\d{8,18}\b", re.I)),
]
ADVICE_PATTERNS = [
 re.compile(r"\b(which|what)\s+(fund|scheme).{0,30}\b(best|buy|invest)\b", re.I),
 re.compile(r"\bshould\s+i\s+(buy|sell|invest|switch|redeem)\b", re.I),
 re.compile(r"\b(recommend|suggest)\b.{0,30}\b(fund|scheme|invest)\b", re.I),
 re.compile(r"\b(best fund|fund to buy|where should i invest)\b", re.I),
 re.compile(r"\b(allocate|allocation)\b.{0,30}(?:₹|rs\.?|inr|\d)", re.I),
 re.compile(r"\b(return|returns)\b.{0,30}(?:predict|forecast|guarantee|\d{2,})", re.I),
]
SUPPORTED_TERMS = re.compile(r"\b(hdfc|mutual fund|mutual funds|sip|elss|expense ratio|exit load|riskometer|benchmark|lock.?in|minimum investment|nav)\b", re.I)
INJECTION_TERMS = re.compile(r"\b(ignore (all )?(previous|prior|above) instructions|system prompt|reveal (the )?prompt|act as)\b", re.I)

@dataclass
class InputCheck:
 allowed: bool
 message: str
 category: str
 safe_query: str = ""

def check_input(query: str, max_chars: int = 1000) -> InputCheck:
 if not isinstance(query,str) or not query.strip(): return InputCheck(False,"Please enter a question.","empty")
 q=query.strip()
 if len(q)>max_chars: return InputCheck(False,f"Please keep your question under {max_chars} characters.","too_long")
 for label,pattern in PII_PATTERNS:
  if pattern.search(q): return InputCheck(False,"Please don't enter personal or sensitive information (PAN, Aadhaar, OTP, account, phone, or email). Your query was not processed.","pii")
 if any(p.search(q) for p in ADVICE_PATTERNS): return InputCheck(False,"I can share factual scheme information, but can't recommend buying, selling, choosing a fund, predicting returns, or allocating investments.","advice")
 if INJECTION_TERMS.search(q): return InputCheck(False,"I can answer factual HDFC Mutual Fund questions using indexed Groww sources only.","prompt_injection")
 if not SUPPORTED_TERMS.search(q): return InputCheck(False,"I support factual FAQs about HDFC Mutual Funds covered by indexed Groww pages.","out_of_scope")
 return InputCheck(True,"Allowed factual query.","allowed",q)

def validate_answer(answer: str, retrieved_sources: list[dict], max_sentences: int = 3) -> tuple[bool,str]:
 if not isinstance(answer,str) or not answer.strip(): return False,"empty_answer"
 sentences=[s for s in re.split(r"(?<=[.!?])\s+",answer.strip()) if s.strip()]
 if len(sentences)>max_sentences: return False,"too_many_sentences"
 urls=[s.get("url","") for s in retrieved_sources if s.get("url")]
 if not urls or not any(url in answer for url in urls): return False,"missing_or_invalid_citation"
 return True,"valid"
