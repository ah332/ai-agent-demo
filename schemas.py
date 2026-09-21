from pydantic import BaseModel,Field

class WeatherArgs(BaseModel):
    city:str = Field(description="城市名称，例如：北京、上海")

class CalculateArgs(BaseModel):
    expression: str = Field(description="数学表达式，例如：1+1、100*5")