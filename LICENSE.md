# License

Copyright (c) 2026 sfw. All rights reserved.

This repository and its contents are made available for viewing and reference only.

No part of this code, documentation, prompts, or other material may be:

- copied, reproduced, or redistributed in whole or in part,
- modified or used to create derivative works,
- used commercially or incorporated into other products or services,
- re-hosted, mirrored, or published elsewhere,

without the prior, express, written permission of the copyright holder.

Viewing this code, cloning it for the purpose of inspection, and running it locally for personal evaluation are permitted. Any other use requires explicit written permission.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY ARISING FROM USE OF THIS SOFTWARE.

---

## Third-party dependencies

CuriosityEngine depends on the following Python packages, each distributed under its own license. Their licenses govern their respective code; redistributing CuriosityEngine does not alter their licensing.

| Package | License | Purpose |
|---|---|---|
| [anthropic](https://github.com/anthropics/anthropic-sdk-python) | MIT | Anthropic API client |
| [openai](https://github.com/openai/openai-python) | Apache-2.0 | OpenAI-compat client |
| [httpx](https://github.com/encode/httpx) | BSD-3-Clause | HTTP client |
| [trafilatura](https://github.com/adbar/trafilatura) | Apache-2.0 / GPLv3+ | HTML→text extraction (`web_fetch`) |
| [networkx](https://github.com/networkx/networkx) | BSD-3-Clause | Knowledge graph |
| [fastapi](https://github.com/fastapi/fastapi) | MIT | Web UI framework |
| [uvicorn](https://github.com/encode/uvicorn) | BSD-3-Clause | ASGI server |
| [jinja2](https://github.com/pallets/jinja) | BSD-3-Clause | Template rendering |
| [python-multipart](https://github.com/Kludex/python-multipart) | Apache-2.0 | Form parsing |

The Docker image additionally builds on `python:3.13-slim` (PSF-2.0). Optional packages (`numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`) ship with the image and follow their respective licenses (BSD/MIT family).

Each dependency's source URL above links to its full license text. Installing or running CuriosityEngine implies acceptance of each dependency's license terms.

## Architectural inspirations

CuriosityEngine borrows architectural *patterns* (not source code) from several public research-agent systems. Pattern-borrowing of ideas described in publicly available research / READMEs is not generally subject to license restrictions, but explicit acknowledgment is the right thing to do:

| System | Pattern borrowed | Their license |
|---|---|---|
| [Google AI Co-Scientist](https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/) | Generation/Reflection/Ranking/Evolution agent split; tournament ranking (Phase 6); Evolution agent (Phase 8) | n/a (closed system) |
| [Sakana AI Scientist](https://github.com/SakanaAI/AI-Scientist) | Mutation loop for idea evolution (Phase 8) | Apache-2.0 |
| [Stanford STORM](https://github.com/stanford-oval/storm) | Multi-perspective question generation (Phase 7) | MIT |
| [Tree of Thoughts](https://arxiv.org/abs/2305.10601) | Branching exploration of hypothesis space (Phase 9) | n/a (research paper) |

**No source code from any of these systems has been incorporated into CuriosityEngine.** The borrows are conceptual: the engine's own implementations of these patterns are independent works.

If a maintainer of any of those projects feels CuriosityEngine's acknowledgment or attribution is insufficient, please open an issue and I'll address it.
