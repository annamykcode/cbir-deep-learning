import logging
from tqdm import tqdm

logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s [%(name)s] : %(message)s')


class Searcher:
    """ Represents a Searcher module, that searches an Elasticsearch index given a user query.

    The searcher queries the "features" field (image feature vector) and ranks the documents using the cosine
    similarity. Specifically, we measure the cosine similarity between the query features and the indexed image
    features and add 1.0 so that the result is always positive.

    """

    def __init__(self):
        """ Initialized Searcher object. """
        self.logger = logging.getLogger(__name__)

    def search_index(self, es, name, queries, k):
        """ Searches an Elasticsearch index.

        The image look-up is done using the "features" field and cosine similarity (1.0 is added for positive results)
        is used to measure the relevance between a query image and an indexed image.

        Args:
            es:
                Elasticsearch server, as Elasticsearch object.
            name:
                index name, as string.
            queries:
                image queries, as list of dictionaries.
            k:
                number of top relevant images to be retrieved, as integer.

        Returns:
            top k retrieved documents with respect to the user's query, as a list of dictionaries of the form
            (query_id, images: [id, filename, path, score]).
        """
        results = []
        for query in tqdm(queries):
            # construct query
            query_body = {
                "query": {
                    "script_score": {
                        "query": {
                            "match_all": {}
                        },
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'features') + 1.0",
                            "params": {
                                "query_vector": query["features"]
                            }
                        }
                    }
                }
            }

            result = es.search(index=name, body=query_body, size=k)
            record = {
                'query_id': query["id"],
                'images': []
            }
            for hit in result['hits']['hits']:
                res = {
                    'id': hit["_source"]["id"],
                    'filename': hit["_source"]["filename"],
                    'path': hit["_source"]["path"],
                    'score': hit["_score"]
                }
                record["images"].append(res)

            results.append(record)

        return results
