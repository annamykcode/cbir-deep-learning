import logging
from tqdm import tqdm

logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s [%(name)s] : %(message)s')


class Indexer:
    """ Represents an Indexer module, that builds an Elasticsearch index in a given Elasticsearch server.

    The client can specify the name of the index and various settings, like the number of the shards and the number
    of replicas for each active shard.

    The images to be indexed should consist of the following fields: ("id", "filename", "path", "features").
    The "features" field mentioned, is a features vector with specific size for all the images to be indexed.

    """

    def __init__(self):
        """ Initialized Indexer object. """
        self.logger = logging.getLogger(__name__)

    def create_index(self, es, name, number_of_shards, number_of_replicas, num_features):
        """ Creates an Elasticsearch index.

        The user should specify four parameters for the Elasticsearch index:
            * the Elasticsearch server,
            * the name of the index,
            * the settings of the index, which are the number of shards and the number of replicas
            each primary shard should have,
            * the size of the feature vector of each image that will be indexed.
        For an image to be indexable it needs to follow a specific mapping. Each image is structured
        as follows: ("id", "filename", "path", "features").

        Args:
            es:
                Elasticsearch server, as Elasticsearch object.
            name:
                index name, as string.
            number_of_shards:
                number of primary shards that an index should have (max 1024), as integer.
            number_of_replicas:
                number of replicas each primary shard has, as integer.
            num_features:
                size of feature vector, as integer.
        """
        # delete index if it pre-exists
        if es.indices.exists(index=[name]):
            self.logger.warning(f'Elasticsearch index "{name}" already exists. Deleting ...')
            es.indices.delete(index=[name])

        # index configurations and document mapping
        config = {
            'settings': {
                'index': {
                    'number_of_shards': number_of_shards,
                    'number_of_replicas': number_of_replicas
                }
            },
            'mappings': {
                'properties': {
                    'id': {
                        'type': 'text',
                        'index': False
                    },
                    'filename': {
                        'type': 'text',
                        'index': False
                    },
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

        es.indices.create(index=name, body=config)

    def index_images(self, es, name, images):
        """ Indexes images into an Elasticsearch index.

        Each indexable image must have the structure of an indexable document, which is as follows:
        ("id", "filename", "path", "features").

        Args:
            es:
                Elasticsearch server, as Elasticsearch object.
            name:
                index name, as string.
            images:
                image documents, as a list of dictionaries.
        """
        for image in tqdm(images):
            es.index(index=name, body=image)
