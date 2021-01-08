import yaml
from conda.models.match_spec import MatchSpec
from yaml import CLoader
from conda.models.version impor<t VersionSpec
from functools import partial


def load_environment(environment_file):
    with open(environment_file, "r") as f:
        environment = yaml.load(f, Loader=CLoader)
    return environment


def dump_environment(environment, environment_file):
    with open(environment_file, "w") as f:
        yaml.dump(environment, f)


def proc_tree(dep_list, func):
    spec_list = []
    for dep in dep_list:
        if isinstance(dep, dict):
            dep_dict = dep
            spec = {}
            for key, val_dep_list in dep_dict.items():
                spec[key] = proc_tree(val_dep_list, func)
        else:
            spec = func(dep)
        spec_list.append(spec)
    return spec_list


def parse_dependencies(dep_list):
    spec_list = proc_tree(dep_list, MatchSpec)
    return spec_list


def serialize_dependencies(spec_list):
    dep_list = proc_tree(spec_list, str)
    return dep_list


def version_eq2ge(spec, except_name_list=None):
    if (except_name_list is None) or (spec.name not in except_name_list):
        kwargs = {
            key: value
            for key, value in spec._match_components.items()
            if key == "channel"
        }
        spec = MatchSpec(
            version=VersionSpec(">=" + str(spec.version.matcher_vo)),
            name=spec.name,
            **kwargs
        )
    return spec


def version_strip(spec, except_name_list=None):
    if (except_name_list is None) or (spec.name not in except_name_list):
        kwargs = {
            key: value
            for key, value in spec._match_components.items()
            if key == "channel"
        }
        spec = MatchSpec(name=spec.name, **kwargs)
    return spec


def proc_env_dependencies(environment, func):
    spec_list = parse_dependencies(environment["dependencies"])
    spec_list = proc_tree(spec_list, func)
    environment["dependencies"] = serialize_dependencies(spec_list)
    return environment


def env_depversions_strip(environment, except_name_list=None):
    func = partial(version_strip, except_name_list=except_name_list)
    return proc_env_dependencies(environment, func)


def env_depversions_eq2ge(environment, except_name_list=None):
    func = partial(version_eq2ge, except_name_list=except_name_list)
    return proc_env_dependencies(environment, func)


def conda_yaml_versions_strip(in_yaml, out_yaml, **kwargs):
    environment = load_environment(in_yaml)
    environment = env_depversions_strip(environment, **kwargs)
    dump_environment(environment, out_yaml)


def conda_yaml_versions_eq2ge(in_yaml, out_yaml, **kwargs):
    environment = load_environment(in_yaml)
    environment = env_depversions_eq2ge(environment, **kwargs)
    dump_environment(environment, out_yaml)
