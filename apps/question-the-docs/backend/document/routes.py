from fastapi import APIRouter, Body, Request
from superduperdb import superduper
from superduperdb.db.mongodb.query import Collection

from backend.document.models import Query
from backend.config import settings

document_router = APIRouter(prefix="/document", tags=["docs"])





@document_router.post(
    "/query",
    response_description="Query document database for data to answer prompt",
)
def query_docs(request: Request, query: Query = Body(...)):
    db = superduper(request.app.mongodb_client.my_database_name)

    context_select = Collection(name="markdown").like({"text": query.query}, n=settings.NEAREST_TO_QUERY, vector_index="documentation_index").find({})
    
    return db.predict('superbot', input=query,  context_select=context_select, context_key='text')
