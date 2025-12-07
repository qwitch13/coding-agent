"""
Code Search Module for Coding Agent
Searches for similar code, solutions, and examples across the web.
"""
import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from urllib.parse import quote_plus, urlencode
import aiohttp

from config import config, CACHE_DIR


@dataclass
class CodeSearchResult:
    """Represents a code search result."""
    source: str  # github, stackoverflow, google, etc.
    title: str
    url: str
    code: str = ""
    language: str = ""
    relevance: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "title": self.title,
            "url": self.url,
            "code": self.code[:500] if self.code else "",
            "language": self.language,
            "relevance": self.relevance,
            "metadata": self.metadata
        }


class SearchCache:
    """Simple file-based cache for search results."""

    def __init__(self, ttl_seconds: int = 3600):
        self.cache_dir = CACHE_DIR / "search_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = ttl_seconds

    def _get_cache_key(self, query: str, source: str) -> str:
        """Generate cache key from query and source."""
        content = f"{query}:{source}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, query: str, source: str) -> Optional[List[Dict]]:
        """Get cached results if valid."""
        key = self._get_cache_key(query, source)
        cache_file = self.cache_dir / f"{key}.json"

        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)

                if time.time() - data.get("timestamp", 0) < self.ttl:
                    return data.get("results", [])
            except:
                pass

        return None

    def set(self, query: str, source: str, results: List[Dict]):
        """Cache results."""
        key = self._get_cache_key(query, source)
        cache_file = self.cache_dir / f"{key}.json"

        data = {
            "query": query,
            "source": source,
            "timestamp": time.time(),
            "results": results
        }

        with open(cache_file, 'w') as f:
            json.dump(data, f)


class GitHubCodeSearch:
    """Search code on GitHub."""

    def __init__(self):
        self.base_url = "https://api.github.com"
        self.session_headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "CodingAgent/1.0"
        }

    async def search(self, query: str, language: Optional[str] = None,
                     max_results: int = 20) -> List[CodeSearchResult]:
        """
        Search GitHub for code.

        Args:
            query: Search query
            language: Filter by programming language
            max_results: Maximum number of results

        Returns:
            List of CodeSearchResult
        """
        results = []

        # Build search query
        search_query = query
        if language:
            search_query += f" language:{language}"

        params = {
            "q": search_query,
            "per_page": min(max_results, 100),
            "sort": "stars",
            "order": "desc"
        }

        try:
            # Search repositories first
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/search/repositories",
                    headers=self.session_headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        for i, repo in enumerate(data.get("items", [])[:max_results]):
                            results.append(CodeSearchResult(
                                source="github",
                                title=repo.get("full_name", ""),
                                url=repo.get("html_url", ""),
                                code="",  # Would need additional API call for code
                                language=repo.get("language", ""),
                                relevance=1.0 - (i / max_results),
                                metadata={
                                    "stars": repo.get("stargazers_count", 0),
                                    "forks": repo.get("forks_count", 0),
                                    "description": repo.get("description", ""),
                                    "topics": repo.get("topics", []),
                                    "updated_at": repo.get("updated_at", "")
                                }
                            ))

                # Also search code directly
                async with session.get(
                    f"{self.base_url}/search/code",
                    headers=self.session_headers,
                    params={"q": search_query, "per_page": min(max_results, 30)},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        for i, item in enumerate(data.get("items", [])[:max_results]):
                            results.append(CodeSearchResult(
                                source="github_code",
                                title=f"{item.get('repository', {}).get('full_name', '')}/{item.get('name', '')}",
                                url=item.get("html_url", ""),
                                code="",
                                language=item.get("name", "").split(".")[-1] if "." in item.get("name", "") else "",
                                relevance=0.8 - (i / max_results * 0.5),
                                metadata={
                                    "path": item.get("path", ""),
                                    "repository": item.get("repository", {}).get("full_name", "")
                                }
                            ))

        except Exception as e:
            print(f"GitHub search error: {e}")

        return results


class StackOverflowSearch:
    """Search code on StackOverflow."""

    def __init__(self):
        self.base_url = "https://api.stackexchange.com/2.3"

    async def search(self, query: str, tags: Optional[List[str]] = None,
                     max_results: int = 20) -> List[CodeSearchResult]:
        """
        Search StackOverflow for answers with code.

        Args:
            query: Search query
            tags: Filter by tags (e.g., ["python", "sorting"])
            max_results: Maximum number of results

        Returns:
            List of CodeSearchResult
        """
        results = []

        params = {
            "order": "desc",
            "sort": "relevance",
            "intitle": query,
            "site": "stackoverflow",
            "pagesize": min(max_results, 50),
            "filter": "withbody"  # Include answer body
        }

        if tags:
            params["tagged"] = ";".join(tags)

        try:
            async with aiohttp.ClientSession() as session:
                # Search questions
                async with session.get(
                    f"{self.base_url}/search/advanced",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()

                        for i, question in enumerate(data.get("items", [])[:max_results]):
                            # Extract code from body
                            body = question.get("body", "")
                            code_blocks = re.findall(r'<code>(.*?)</code>', body, re.DOTALL)
                            code = "\n".join(code_blocks[:3]) if code_blocks else ""

                            # Clean HTML
                            code = re.sub(r'<[^>]+>', '', code)

                            results.append(CodeSearchResult(
                                source="stackoverflow",
                                title=question.get("title", ""),
                                url=question.get("link", ""),
                                code=code[:1000],
                                language=question.get("tags", [""])[0] if question.get("tags") else "",
                                relevance=1.0 - (i / max_results),
                                metadata={
                                    "score": question.get("score", 0),
                                    "answer_count": question.get("answer_count", 0),
                                    "is_answered": question.get("is_answered", False),
                                    "tags": question.get("tags", []),
                                    "view_count": question.get("view_count", 0)
                                }
                            ))

        except Exception as e:
            print(f"StackOverflow search error: {e}")

        return results


class WebSearch:
    """General web search for code using DuckDuckGo."""

    def __init__(self):
        self.base_url = "https://html.duckduckgo.com/html/"

    async def search(self, query: str, code_only: bool = True,
                     max_results: int = 20) -> List[CodeSearchResult]:
        """
        Search the web for code-related content.

        Args:
            query: Search query
            code_only: Add code-related terms to query
            max_results: Maximum number of results

        Returns:
            List of CodeSearchResult
        """
        results = []

        # Add code-related terms if requested
        search_query = query
        if code_only and not any(term in query.lower() for term in ['code', 'example', 'implementation', 'github']):
            search_query = f"{query} code example"

        try:
            from bs4 import BeautifulSoup

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url,
                    data={"q": search_query},
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, "lxml")

                        for i, result_div in enumerate(soup.select(".result")[:max_results]):
                            title_elem = result_div.select_one(".result__a")
                            snippet_elem = result_div.select_one(".result__snippet")

                            if title_elem:
                                title = title_elem.get_text(strip=True)
                                url = title_elem.get("href", "")

                                # Extract actual URL from DuckDuckGo redirect
                                if "uddg=" in url:
                                    from urllib.parse import unquote
                                    url_match = re.search(r"uddg=([^&]+)", url)
                                    if url_match:
                                        url = unquote(url_match.group(1))

                                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                                # Determine source from URL
                                source = "web"
                                if "github.com" in url:
                                    source = "github"
                                elif "stackoverflow.com" in url:
                                    source = "stackoverflow"
                                elif "docs." in url or "documentation" in url.lower():
                                    source = "documentation"

                                results.append(CodeSearchResult(
                                    source=source,
                                    title=title,
                                    url=url,
                                    code="",  # Would need to fetch page for code
                                    relevance=1.0 - (i / max_results),
                                    metadata={
                                        "snippet": snippet
                                    }
                                ))

        except Exception as e:
            print(f"Web search error: {e}")

        return results


class CodeSimilaritySearch:
    """Find similar code across the web."""

    def __init__(self):
        self.github = GitHubCodeSearch()
        self.stackoverflow = StackOverflowSearch()

    def _extract_keywords(self, code: str) -> List[str]:
        """Extract keywords from code for searching."""
        # Remove comments
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

        # Remove strings
        code = re.sub(r'"[^"]*"', '', code)
        code = re.sub(r"'[^']*'", '', code)

        # Extract identifiers
        identifiers = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', code)

        # Filter common keywords
        common = {'def', 'class', 'if', 'else', 'for', 'while', 'return', 'import',
                  'from', 'as', 'try', 'except', 'with', 'in', 'is', 'not', 'and',
                  'or', 'True', 'False', 'None', 'self', 'function', 'var', 'let',
                  'const', 'async', 'await', 'public', 'private', 'static', 'void'}

        keywords = [id for id in identifiers if id.lower() not in common and len(id) > 2]

        # Count frequency and return top keywords
        from collections import Counter
        keyword_counts = Counter(keywords)
        return [kw for kw, _ in keyword_counts.most_common(5)]

    def _detect_language(self, code: str) -> str:
        """Detect programming language from code."""
        indicators = {
            'python': [r'\bdef\s+\w+\s*\(', r'\bimport\s+\w+', r'\bself\b', r':\s*$'],
            'javascript': [r'\bfunction\s+\w+', r'\bconst\s+\w+', r'\blet\s+\w+', r'=>'],
            'typescript': [r':\s*\w+\[\]', r'interface\s+\w+', r'type\s+\w+\s*='],
            'java': [r'\bpublic\s+class', r'\bprivate\s+\w+', r'System\.out'],
            'go': [r'\bfunc\s+\w+', r'\bpackage\s+\w+', r':='],
            'rust': [r'\bfn\s+\w+', r'\blet\s+mut\b', r'\bimpl\s+\w+'],
            'cpp': [r'#include\s*<', r'\bstd::', r'\btemplate\s*<'],
        }

        scores = {}
        for lang, patterns in indicators.items():
            score = sum(1 for p in patterns if re.search(p, code, re.MULTILINE))
            if score > 0:
                scores[lang] = score

        return max(scores, key=scores.get) if scores else ""

    async def find_similar(self, code: str, max_results: int = 20) -> List[CodeSearchResult]:
        """
        Find similar code across multiple sources.

        Args:
            code: Code snippet to find similar code for
            max_results: Maximum number of results

        Returns:
            List of CodeSearchResult ordered by relevance
        """
        # Extract keywords and detect language
        keywords = self._extract_keywords(code)
        language = self._detect_language(code)

        if not keywords:
            return []

        # Build search query
        query = " ".join(keywords)

        # Search multiple sources in parallel
        results = []

        github_task = self.github.search(query, language=language, max_results=max_results // 2)
        so_task = self.stackoverflow.search(query, tags=[language] if language else None, max_results=max_results // 2)

        github_results, so_results = await asyncio.gather(github_task, so_task)

        results.extend(github_results)
        results.extend(so_results)

        # Sort by relevance
        results.sort(key=lambda r: r.relevance, reverse=True)

        return results[:max_results]


class CodeSearchManager:
    """
    Main code search manager.
    Coordinates searches across multiple sources.
    """

    def __init__(self):
        self.github = GitHubCodeSearch()
        self.stackoverflow = StackOverflowSearch()
        self.web = WebSearch()
        self.similarity = CodeSimilaritySearch()
        self.cache = SearchCache()

    async def search_github(self, query: str, language: Optional[str] = None,
                             max_results: int = 20) -> List[CodeSearchResult]:
        """Search GitHub for code."""
        # Check cache
        cached = self.cache.get(query, f"github:{language}")
        if cached:
            return [CodeSearchResult(**r) for r in cached]

        results = await self.github.search(query, language, max_results)

        # Cache results
        self.cache.set(query, f"github:{language}", [r.to_dict() for r in results])

        return results

    async def search_stackoverflow(self, query: str, tags: Optional[List[str]] = None,
                                     max_results: int = 20) -> List[CodeSearchResult]:
        """Search StackOverflow for answers."""
        tag_str = ",".join(tags) if tags else ""
        cached = self.cache.get(query, f"stackoverflow:{tag_str}")
        if cached:
            return [CodeSearchResult(**r) for r in cached]

        results = await self.stackoverflow.search(query, tags, max_results)

        self.cache.set(query, f"stackoverflow:{tag_str}", [r.to_dict() for r in results])

        return results

    async def search_web(self, query: str, max_results: int = 20) -> List[CodeSearchResult]:
        """Search the web for code."""
        cached = self.cache.get(query, "web")
        if cached:
            return [CodeSearchResult(**r) for r in cached]

        results = await self.web.search(query, max_results=max_results)

        self.cache.set(query, "web", [r.to_dict() for r in results])

        return results

    async def search_all(self, query: str, language: Optional[str] = None,
                          max_results: int = 20) -> List[CodeSearchResult]:
        """
        Search all sources for code.

        Args:
            query: Search query
            language: Filter by programming language
            max_results: Maximum total results

        Returns:
            Combined and sorted list of CodeSearchResult
        """
        # Search all sources in parallel
        github_task = self.search_github(query, language, max_results // 3)
        so_task = self.search_stackoverflow(query, [language] if language else None, max_results // 3)
        web_task = self.search_web(query, max_results // 3)

        github_results, so_results, web_results = await asyncio.gather(
            github_task, so_task, web_task
        )

        # Combine and deduplicate by URL
        all_results = []
        seen_urls = set()

        for result in github_results + so_results + web_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                all_results.append(result)

        # Sort by relevance
        all_results.sort(key=lambda r: r.relevance, reverse=True)

        return all_results[:max_results]

    async def find_similar_code(self, code: str, max_results: int = 20) -> List[CodeSearchResult]:
        """
        Find similar code to the provided snippet.

        Args:
            code: Code snippet to find similar code for
            max_results: Maximum number of results

        Returns:
            List of similar code results
        """
        return await self.similarity.find_similar(code, max_results)

    async def search_for_solution(self, error: str, code_context: str = "",
                                    language: str = "") -> List[CodeSearchResult]:
        """
        Search for solutions to an error.

        Args:
            error: Error message
            code_context: Code that caused the error
            language: Programming language

        Returns:
            List of potential solutions
        """
        # Clean error message
        error_clean = re.sub(r'File "[^"]+", line \d+', '', error)
        error_clean = re.sub(r'\s+', ' ', error_clean).strip()

        # Build query
        query = f"{language} {error_clean[:100]}" if language else error_clean[:100]

        # Search StackOverflow primarily (best for errors)
        so_results = await self.search_stackoverflow(
            query,
            tags=[language] if language else None,
            max_results=10
        )

        # Also search web
        web_results = await self.search_web(f"{query} solution fix", max_results=10)

        # Combine and prioritize StackOverflow results
        all_results = so_results + web_results

        # Boost results that mention the error type
        error_type_match = re.search(r'(\w+Error|\w+Exception)', error)
        if error_type_match:
            error_type = error_type_match.group(1)
            for result in all_results:
                if error_type.lower() in result.title.lower():
                    result.relevance *= 1.5

        all_results.sort(key=lambda r: r.relevance, reverse=True)

        return all_results[:20]


# Global search manager instance
search_manager = CodeSearchManager()


# Convenience functions
async def search_code(query: str, language: Optional[str] = None,
                       source: str = "all", max_results: int = 20) -> List[CodeSearchResult]:
    """
    Search for code across the web.

    Args:
        query: Search query
        language: Filter by programming language
        source: Search source (github, stackoverflow, web, all)
        max_results: Maximum number of results

    Returns:
        List of CodeSearchResult
    """
    if source == "github":
        return await search_manager.search_github(query, language, max_results)
    elif source == "stackoverflow":
        return await search_manager.search_stackoverflow(
            query, [language] if language else None, max_results
        )
    elif source == "web":
        return await search_manager.search_web(query, max_results)
    else:
        return await search_manager.search_all(query, language, max_results)


async def find_similar(code: str, max_results: int = 20) -> List[CodeSearchResult]:
    """Find similar code to the provided snippet."""
    return await search_manager.find_similar_code(code, max_results)


async def search_solution(error: str, code: str = "", language: str = "") -> List[CodeSearchResult]:
    """Search for solutions to an error."""
    return await search_manager.search_for_solution(error, code, language)
