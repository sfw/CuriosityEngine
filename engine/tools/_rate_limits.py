"""Rate limiters for throttled external endpoints.

One instance per endpoint, shared process-wide. Every network-touching tool
calls `.acquire()` on the appropriate limiter before firing. Serial runs gain
pacing (previously the LLM could burst 10+ tool_uses in one response with
zero delay); parallel runs gain the hard guarantee that we don't blow past
public API budgets no matter how many engine threads are active.

Rates are sized for politeness, not for peak legal throughput:

- arXiv's user manual asks for 3s between requests. We enforce that exactly.
- Semantic Scholar unauthenticated quota is 100 req / 5 min = 0.33 req/s.
  We pace at 1/3s to stay well clear of bursty depletion.
- Crossref "polite pool" is generous (50+ req/s with a mailto header); we
  still cap at 5 req/s to be conservative.
- DuckDuckGo/Bing are scraping endpoints — no official rate; 1 req/2s is
  the empirically-safe pace that our existing _PaceGate used.
- web_fetch is generic HTTP — per-host 3 req/s so we don't hammer any one
  paper server. arxiv.org + paper repositories benefit from the same
  per-host discipline.
- Internet Archive + Wikimedia are rate-friendly but we still pace to
  respect shared infrastructure.

If you're seeing frequent "rate limited" tool errors in the log, raise the
burst (more tolerance for clusters) before raising the rate. Raising the rate
risks actual blocks; raising burst just lets short spikes through.

Every limiter also carries a `jitter` — max seconds of uniform-random delay
added AFTER token acquisition. Without jitter, every request from this engine
lands on the same fixed interval (exactly 3.00s between arXiv calls, etc.),
which is a trivial bot fingerprint. With jitter, pacing becomes 3.0–4.0s
(or whatever window), breaking the regularity. Jitter is generous on
throttle-prone endpoints (arxiv, semantic_scholar) and minimal on generous
ones (crossref).
"""

from __future__ import annotations

from engine.tools.base import HostRateLimiter, RateLimiter

# --- Academic endpoints --------------------------------------------------------
# arXiv: documented as "1 req / 3s" in their user manual, but empirical
# evidence shows that pacing reliably triggers HTTP 429 on sustained
# burst workloads (gap scan verification fires 80+ probes back-to-back).
# Their server applies sliding-window detection in addition to the
# documented limit. Slowed to 1/5s + 2s jitter (5–7s pacing, ~10 req/min)
# to stay clear of the burst-detection threshold. Combined with
# cooldown-on-429 in academic_search.py, this is belt-and-suspenders:
# the slower steady state reduces 429 frequency, and any 429 that does
# fire engages a 60s cooldown automatically.
ARXIV = RateLimiter(rate=1 / 5.0, burst=1, jitter=2.0, name="arxiv")

# Semantic Scholar unauthenticated: documented 100 req / 5min ≈ 0.33 req/s.
# Same story as arxiv — empirical 429s at 1/3s pacing during bursts
# despite being below the documented limit. Slowed to 1/5s + 2s jitter
# to match arxiv. Cooldown-on-429 backs it off when their throttle fires.
SEMANTIC_SCHOLAR = RateLimiter(rate=1 / 5.0, burst=1, jitter=2.0, name="semantic_scholar")

# Crossref polite pool is 50+ req/s; we cap at 5 for safety + burst=10 so a
# whole academic_search call can fire in one shot. Small jitter — this endpoint
# is generous, we just want to break perfect uniformity.
CROSSREF = RateLimiter(rate=5.0, burst=10, jitter=0.2, name="crossref")

# --- Web-search scrapers -------------------------------------------------------
# Note: engine/tools/web_search.py has its own per-host `_PaceGate` that
# handles DuckDuckGo and Bing. _PaceGate enforces the same ~2s interval AND
# implements cooldown-on-429 (a feature the generic RateLimiter doesn't have).
# Leaving that tool-local; the limiters below are unused until we unify.
DUCKDUCKGO = RateLimiter(rate=0.5, burst=1, jitter=1.0, name="duckduckgo")  # (unused — see web_search._PaceGate)
BING = RateLimiter(rate=0.5, burst=1, jitter=1.0, name="bing")              # (unused — see web_search._PaceGate)

# --- Generic HTTP fetching -----------------------------------------------------
# web_fetch is per-host so busy hosts (e.g. arxiv.org for bulk paper fetches)
# don't get hammered, while other hosts remain snappy.
# Jitter 0.3s — web_fetch already fans out across hosts, but within a host
# we still want a little fuzz.
WEB_FETCH = HostRateLimiter(rate=3.0, burst=5, jitter=0.3, name="web_fetch")

# --- Archive / reference endpoints --------------------------------------------
# Internet Archive + Wikimedia Commons + Openverse — polite defaults.
# Jitter is a modest fraction of the nominal interval.
ARCHIVE_ORG = RateLimiter(rate=2.0, burst=4, jitter=0.5, name="archive.org")
WIKIMEDIA = RateLimiter(rate=3.0, burst=5, jitter=0.3, name="wikimedia")
OPENVERSE = RateLimiter(rate=2.0, burst=4, jitter=0.5, name="openverse")
