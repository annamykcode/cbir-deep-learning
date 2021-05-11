from elasticsearch import Elasticsearch


class ImageIndexer:
    """ An image indexer using Elasticsearch. """

    def __init__(self, hosts):
        """ Initializes Elasticsearch object.

        Args:
            hosts:
                List of hosts where the Elasticsearch will run.
        """
        self._es = Elasticsearch(hosts=hosts, timeout=30, retry_on_timeout=True)

    def create_index(self, index_name, num_features):
        """ Create an Elasticsearch index with the given name.

        The image files that are indexed consist of two fields:
            * path: the file path of the image file (not indexed)
            * features: feature vector of image.

        Args:
            index_name:
                Index name, as a string.
            num_features:
                Number of features, as a integer.
        """
        # delete index if it pre-exists
        if self._es.indices.exists(index=index_name):
            self._es.indices.delete(index=index_name)

        config = {
            'settings': {
                'index': {
                    'number_of_shards': 10,
                    'number_of_replicas': 0
                }
            },
            'mappings': {
                'properties': {
                    'path': {
                        'type': 'text',
                        'index': False
                    },
                    'features': {
                        'type': 'dense_vector',
                        'dims': num_features,
                    }
                }
            }
        }

        self._es.indices.create(index=index_name, body=config)

    def index_docs(self, index_name, dataset):
        """ Index image files using Elasticsearch.

        Args:
            index_name:
                Index name, as a string.
            dataset:
                List of dictionaries, with the fields: path (string), features (vector)
        """
        total = 1
        for data in dataset:
            self._es.index(index=index_name, body=data)

            if total % 1000 == 0:
                print(f'[INFO] indexer.ImageIndexer.index_docs - {total} images have been indexed')

            total += 1

    def query_index(self, index_name, features):
        """ Query Elasticsearch index.

        Args:
            index_name:
                Index name, as a string.
            features:
                Feature vector, as a numpy array.

        Returns:
            The results of the query, as returned by Elasticsearch   .
        """
        query = {
            'query': {
                'script_score': {
                    'query': {
                        'match_all': {}
                    },
                    'script': {
                        'source': "cosineSimilarity(params.query_vector, 'features') + 1.0",
                        'params': {'query_vector': features}
                    }
                }
            }
        }

        res = self._es.search(index=index_name, body=query)

        return res
