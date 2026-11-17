from pydantic import BaseModel, Field
from typing import Annotated

class User(BaseModel):
    id: int = Annotated[int, Field(gt=0)]
    name: str = Annotated[str, Field(min_length=1)]