from pydantic import BaseModel, Field
from typing import Annotated


class User(BaseModel):
    """Class to define the current user of the app

    Args:
        BaseModel (User): pydantic basemodel to enforce limits
    """

    id: Annotated[int, Field(gt=0)]
    name: Annotated[str, Field(min_length=1)]
