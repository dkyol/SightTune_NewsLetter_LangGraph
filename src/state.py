import operator
from typing import Optional

from langchain_core.messages import AnyMessage
from typing_extensions import Annotated, List, Literal, TypedDict

Route = Literal["researcher", "writer", "reviewer", "newsletter_compiler"]

class NewsletterState(TypedDict):
    theme:             str
    topics:            List[str]
    current_index:     int
    current_research:  Optional[str]
    current_draft:     Optional[str]
    revision_count:    int
    articles:          List[str]
    research_messages: List[AnyMessage]
    messages:          Annotated[List[AnyMessage], operator.add]
    output:            Optional[str]
