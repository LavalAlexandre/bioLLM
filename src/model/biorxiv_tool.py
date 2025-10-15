from agents import function_tool
from ddgs import DDGS


@function_tool
def search_biorxiv(query: str) -> str:
    """
    Search bioRxiv for biology preprints using DuckDuckGo.

    Args:
        query: The search query for bioRxiv preprints

    Returns:
        Search results as a formatted string
    """
    try:
        # Add site restriction for bioRxiv
        search_query = f"site:biorxiv.org {query}"

        # Initialize DuckDuckGo search
        ddgs = DDGS()

        # Perform search and get top 5 results
        results = list(ddgs.text(search_query, max_results=5))

        if not results:
            return f"No bioRxiv preprints found for query: {query}"

        # Format results
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(
                f"{i}. {result.get('title', 'No title')}\n"
                f"   URL: {result.get('href', 'No URL')}\n"
                f"   {result.get('body', 'No description')}"
            )

        return "\n\n".join(formatted_results)

    except Exception as e:
        return f"Error searching bioRxiv: {str(e)}"


# The decorator creates the tool, assign it to a variable
BiorxivSearchTool = search_biorxiv
