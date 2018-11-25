###
# Custom command to use with click a config file
# source: https://stackoverflow.com/questions/
#   46358797/python-click-supply-arguments-and
#   -options-from-a-configuration-file
###
import click
import yaml


def CommandWithConfigFile(config_file_param_name):

    class CustomCommandClass(click.Command):

        def invoke(self, ctx):
            config_file = ctx.params[config_file_param_name]
            if config_file is not None:
                with open(config_file) as f:
                    config_data = yaml.load(f)
                if config_data is not None:
                    assert type(config_data) is dict
                    for param, value in config_data.items():
                        if param in ctx.params.keys():
                            ctx.params[param] = config_data[param]

            return super(CustomCommandClass, self).invoke(ctx)

    return CustomCommandClass
