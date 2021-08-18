import os

from everett.component import ConfigOptions, RequiredConfigMixin
from everett.ext.yamlfile import ConfigYamlEnv
from everett.manager import ConfigManager, ConfigOSEnv, ListOf


class MinioConfig(RequiredConfigMixin):
    """Contains all MINIO information"""
    required_config = ConfigOptions()

    required_config.add_option(
        'host',
        parser=str,
        default="localhost",
        doc='Host of the minio installation'
    )

    required_config.add_option(
        'port',
        parser=int,
        default="9000",
        doc='Port of the minio installation'
    )

    required_config.add_option(
        'key',
        parser=str,
        default="abcdACCESS",
        doc='Key used to connect to minio'
    )

    required_config.add_option(
        'secret',
        parser=str,
        default="abcdSECRET",
        doc='Secret used to connect to minio'
    )

    required_config.add_option(
        'upload_bucket',
        parser=str,
        default='upload',
        doc='The bucket documents are uploaded to per default'
    )

    def __init__(self, config):
        self.config = config.with_options(self)
        self.host = self.config('host')
        self.port = self.config('port')
        self.key = self.config('key')
        self.secret = self.config('secret')
        self.upload_bucket = self.config('upload_bucket')

    def __call__(self, *args, **kwargs):
        return self.config(*args, **kwargs)


class DatabaseConfig(RequiredConfigMixin):
    required_config = ConfigOptions()

    required_config.add_option(
        'host',
        parser=str,
        default='localhost',
        doc='The host url of the database'
    )

    required_config.add_option(
        'port',
        parser=str,
        default="3308",
        doc='The database port'
    )

    required_config.add_option(
        'user',
        parser=str,
        default='root',
        doc='The database user'
    )

    required_config.add_option(
        'password',
        parser=str,
        default="root",
        doc='The database password'
    )

    required_config.add_option(
        'schema',
        parser=str,
        default="GEN_NER_UI",
        doc='The database schema'
    )

    def __init__(self, config):
        self.config = config.with_options(self)

        self.host = self.config("host")
        self.port = self.config("port")
        self.user = self.config("user")
        self.password = self.config("password")
        self.schema = self.config("schema")

    def __call__(self, *args, **kwargs):
        return self.config(*args, **kwargs)


class AmqpConfig(RequiredConfigMixin):
    """Contains all AMQP information"""
    required_config = ConfigOptions()

    required_config.add_option(
        'url',
        parser=str,
        doc='AMQP connection str used to connect to broker.',
        default='amqp://amqp_user:amqp_pass@localhost:5672/nerui?heartbeat=30'
    )

    def __init__(self, config):
        self.config = config.with_options(self)
        self.url = self.config('url')

    def __call__(self, *args, **kwargs):
        return self.config(*args, **kwargs)


class ImageConfig(RequiredConfigMixin):
    """Contains all AMQP information"""
    required_config = ConfigOptions()

    required_config.add_option(
        'width',
        parser=int,
        doc='width the images in the view get resized to',
        default='1200'
    )

    required_config.add_option(
        'height',
        parser=int,
        doc='height the images in the view get resized to',
        default='1200'
    )

    def __init__(self, config):
        self.config = config.with_options(self)
        self.width = self.config('width')
        self.height = self.config('height')

    def __call__(self, *args, **kwargs):
        return self.config(*args, **kwargs)


class PipelineConfig(RequiredConfigMixin):
    required_config = ConfigOptions()

    required_config.add_option(
        'queue_name',
        parser=str,
        default="ocr_pipeline",
        doc='The name of the rabbitmq queue'
    )

    required_config.add_option(
        'name',
        parser=str,
        default="ocr_pipeline",
        doc="The display name of the pipeline"
    )

    required_config.add_option(
        'depends_on',
        parser=ListOf(str),
        default="",
        doc="Describe what inputs are required from other pipelines"
    )

    required_config.add_option(
        'return_pages',
        parser=bool,
        default="True",
        doc="Determines whether the pipeline splits a document into pages"
    )

    def __init__(self, config):
        self.config = config.with_options(self)
        self.queue_name = self.config('queue_name')
        self.name = self.config('name')
        self.depends_on = self.config('depends_on')
        self.return_pages = self.config('return_pages')

    def __call__(self, *args, **kwargs):
        return self.config(*args, **kwargs)


class PipelinesConfig(RequiredConfigMixin):
    required_config = ConfigOptions()

    required_config.add_option(
        'count',
        parser=int,
        doc='The count of pipelines to be used',
        default="1"
    )

    required_config.add_option(
        'return_last',
        parser=bool,
        doc='determines weather only the last pipeline result should be used',
        default="True"
    )

    def __init__(self, config):
        self.config = config.with_options(self)
        self.count = self.config('count')
        self.return_last = self.config('return_last')
        self.pipelines = [PipelineConfig(self.config.with_namespace(f"P{i}"))
                          for i in range(self.count)]

    def __call__(self, *args, **kwargs):
        return self.config(*args, **kwargs)


class EntityGroupConfig(RequiredConfigMixin):
    required_config = ConfigOptions()

    required_config.add_option(
        'name',
        parser=str,
        default="Organisations and other",
        doc="The display name of the entity group"
    )

    required_config.add_option(
        'entities',
        parser=ListOf(str),
        default="ORG,MISC",
        doc="Describe what inputs are required from other pipelines"
    )

    def __init__(self, config):
        self.config = config.with_options(self)
        self.name = self.config('name')
        self.entities = self.config('entities')

    def __call__(self, *args, **kwargs):
        return self.config(*args, **kwargs)


class EntityGroupsConfig(RequiredConfigMixin):
    required_config = ConfigOptions()

    required_config.add_option(
        'count',
        parser=int,
        doc='The count of entity-groups to be used',
        default="1"
    )

    def __init__(self, config):
        self.config = config.with_options(self)
        self.count = self.config('count')
        self.groups = [EntityGroupConfig(self.config.with_namespace(f"E{i}"))
                       for i in range(self.count)]

    def __call__(self, *args, **kwargs):
        return self.config(*args, **kwargs)


class TargetServerConfig(RequiredConfigMixin):
    required_config = ConfigOptions()

    required_config.add_option(
        'target_server_url',
        parser=str,
        doc='target server api for notification sending',
        default="http://127.0.0.1:7000/bow_notification"
    )

    def __init__(self, config):
        self.config = config.with_options(self)
        self.target_server_url = self.config('target_server_url')

    def __call__(self, *args, **kwargs):
        return self.config(*args, **kwargs)


class Config(object):
    def __init__(self):
        self.manager = ConfigManager(
            environments=[
                ConfigOSEnv(),
                ConfigYamlEnv([
                    os.environ.get('CONFIG_YAML'),
                    './config.yaml',
                    './config.yml',
                    '/etc/config.yaml'
                    '/etc/config.yml'
                ]),
            ]).with_namespace('config')
        self.minio = MinioConfig(self.manager.with_namespace('minio'))
        self.amqp = AmqpConfig(self.manager.with_namespace('amqp'))
        self.database = DatabaseConfig(self.manager.with_namespace('database'))
        self.pipelines = PipelinesConfig(self.manager
                                         .with_namespace('pipelines'))
        self.entitygroups = EntityGroupsConfig(self.manager
                                               .with_namespace('entitygroups'))
        self.image = ImageConfig(self.manager.with_namespace('image'))
        self.target = TargetServerConfig(self.manager.with_namespace('target'))
