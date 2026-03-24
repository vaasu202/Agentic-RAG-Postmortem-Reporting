from tools.kb_search import search_incident_knowledge_base

q = "router rule caused outage"
hits = search_incident_knowledge_base(q, k=5)

print(f"hits={len(hits)}")
for h in hits:
    c = h["citation"]
    print("-", c["doc_type"], c["filename"], "score=", h["score"])
