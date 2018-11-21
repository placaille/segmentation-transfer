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
                    for param, value in ctx.params.items():
                        if value is None and param in config_data:
                            ctx.params[param] = config_data[param]

            return super(CustomCommandClass, self).invoke(ctx)

    return CustomCommandClass
