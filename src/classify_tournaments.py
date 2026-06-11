QUALIFIER_KEYWORDS  = ["qualification", "qualifier", "qualifying"]

WORLD_CUP_KEYWORDS  = ["fifa world cup"]
 
TOURNAMENT_KEYWORDS = [
    "african cup of nations", "uefa euro", "copa américa", "copa america",
    "gold cup", "confederations cup", "aff championship", "oceania nations cup",
    "afc asian cup", "concacaf nations league", "uefa nations league",
    "eaff championship", "waff championship", "saff cup", "cosafa cup",
    "cafa nations cup", "arab cup", "gulf cup", "asean championship",
]
 
def classify_tournament(tournament: str) -> str:
    t = tournament.lower()
    if any(k in t for k in QUALIFIER_KEYWORDS):
        return "q"
    if any(k in t for k in WORLD_CUP_KEYWORDS):
        return "w"
    if any(k in t for k in TOURNAMENT_KEYWORDS):
        return "t"
    return "f"