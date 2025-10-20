from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from typing import TypedDict,Annotated,Optional,Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)

class MovieInfo(TypedDict):
    title: str
    director: str
    release_year: int
    genre: Literal["Action", "Comedy", "Drama", "Horror", "Sci-Fi", "Romance"]
    rating: Annotated[Optional[float], "IMDb rating out of 10"]
    summary: Annotated[Optional[str], "A brief summary of the movie plot"]

structured_model=model.with_structured_output(MovieInfo)

result= structured_model.invoke("""A tragic incident forces Anirudh, a middle-aged man, to take a trip down memory lane and reminisce his college days along with his friends, who were labelled as losers.

Director
Nitesh Tiwari
Writers
Piyush GuptaNikhil MehrotraNitesh Tiwari
Stars
Sushant Singh RajputShraddha KapoorVarun Sharma.The setup was within 1990's till present where Anni, a divorcee used his past experience in overcoming challenges of being a loser in college where he met His Wife, Maya and his Losers friends, Saxa, Mummy, Acid, Derek and Bevda. His shares past experience to his son, Raghav who struggles with being failed at getting an offer to college despite being an excellent student.—rynhalliwell
 Release date
September 6, 2019 (United States)
Country of origin
India
Official sites
Official site (Japan)Stream Chhichhore officially on Hotstar Singapore
Language
Hindi
Also known as
Flippant
Filming locations
Westminster, London, England, UK(location)
Production companies
Fox STAR StudiosFox STAR StudiosNadiadwala Grandson Entertainment
See more company credits at IMDbPro
Box office
Gross US & Canada
$2,004,400
Opening weekend US & Canada
$614,335Sep 8, 2019
Gross worldwide
$3,311,391""")

print(result['summary'])
