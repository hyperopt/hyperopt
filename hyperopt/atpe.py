"""
    Implements the ATPE algorithm. See
    https://www.electricbrain.io/blog/learning-to-optimize
    and
    https://www.electricbrain.io/blog/optimizing-optimization to learn more
"""

__authors__ = "Bradley Arsenault"
__license__ = "3-clause BSD License"
__contact__ = "github.com/hyperopt/hyperopt"

from hyperopt import hp
from contextlib import contextmanager
import re
import functools
import random
import numpy
import numpy.random
import pkg_resources
import tempfile
import scipy.stats
import os
import math
import hyperopt
import datetime
import json
import copy

# Windows doesn't support opening a NamedTemporaryFile.
# Solution inspired in https://stackoverflow.com/a/46501017/147507


@contextmanager
def ClosedNamedTempFile(contents):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            file_name = f.name
            f.write(contents)
        yield file_name
    finally:
        os.unlink(file_name)


class Hyperparameter:
    """This class represents a hyperparameter."""

    def __init__(self, config, parent=None, root="root"):
        self.config = config
        self.root = root
        self.name = root[5:]
        self.parent = parent
        self.resultVariableName = re.sub("\\.\\d+\\.", ".", self.name)

        self.hyperoptVariableName = self.root
        if "name" in config:
            self.hyperoptVariableName = config["name"]

    def createHyperoptSpace(self, lockedValues=None):
        name = self.root

        if lockedValues is None:
            lockedValues = {}

        if "anyOf" in self.config or "oneOf" in self.config:
            v = "anyOf" if "anyOf" in self.config else "oneOf"
            data = self.config[v]

            subSpaces = [
                Hyperparameter(
                    param, self, name + "." + str(index)
                ).createHyperoptSpace(lockedValues)
                for index, param in enumerate(data)
            ]
            for index, space in enumerate(subSpaces):
                space["$index"] = index

            choices = hp.choice(self.hyperoptVariableName, subSpaces)

            return choices
        elif "enum" in self.config:
            if self.name in lockedValues:
                return lockedValues[self.name]

            choices = hp.choice(self.hyperoptVariableName, self.config["enum"])
            return choices
        elif "constant" in self.config:
            if self.name in lockedValues:
                return lockedValues[self.name]

            return self.config["constant"]
        elif self.config["type"] == "object":
            space = {}
            for key in self.config["properties"].keys():
                config = self.config["properties"][key]
                space[key] = Hyperparameter(
                    config, self, name + "." + key
                ).createHyperoptSpace(lockedValues)
            return space
        elif self.config["type"] == "number":
            if self.name in lockedValues:
                return lockedValues[self.name]

            mode = self.config.get("mode", "uniform")
            scaling = self.config.get("scaling", "linear")

            if mode == "uniform":
                min = self.config.get("min", 0)
                max = self.config.get("max", 1)
                rounding = self.config.get("rounding", None)

                if scaling == "linear":
                    if rounding is not None:
                        return hp.quniform(
                            self.hyperoptVariableName, min, max, rounding
                        )
                    else:
                        return hp.uniform(self.hyperoptVariableName, min, max)
                elif scaling == "logarithmic":
                    if rounding is not None:
                        return hp.qloguniform(
                            self.hyperoptVariableName,
                            math.log(min),
                            math.log(max),
                            rounding,
                        )
                    else:
                        return hp.loguniform(
                            self.hyperoptVariableName, math.log(min), math.log(max)
                        )
            if mode == "randint":
                min = self.config.get("min")
                max = self.config.get("max")
                return hp.randint(self.hyperoptVariableName, min, max)

            if mode == "normal":
                mean = self.config.get("mean", 0)
                stddev = self.config.get("stddev", 1)
                rounding = self.config.get("rounding", None)

                if scaling == "linear":
                    if rounding is not None:
                        return hp.qnormal(
                            self.hyperoptVariableName, mean, stddev, rounding
                        )
                    else:
                        return hp.normal(self.hyperoptVariableName, mean, stddev)
                elif scaling == "logarithmic":
                    if rounding is not None:
                        return hp.qlognormal(
                            self.hyperoptVariableName,
                            math.log(mean),
                            math.log(stddev),
                            rounding,
                        )
                    else:
                        return hp.lognormal(
                            self.hyperoptVariableName, math.log(mean), math.log(stddev)
                        )

    def getFlatParameterNames(self):
        name = self.root

        if "anyOf" in self.config or "oneOf" in self.config:
            keys = set()
            v = "anyOf" if "anyOf" in self.config else "oneOf"
            data = self.config[v]

            for index, param in enumerate(data):
                subKeys = Hyperparameter(
                    param, self, name + "." + str(index)
                ).getFlatParameterNames()
                for key in subKeys:
                    keys.add(key)

            return keys
        elif "enum" in self.config or "constant" in self.config:
            return [name]
        elif self.config["type"] == "object":
            keys = set()
            for key in self.config["properties"].keys():
                config = self.config["properties"][key]
                subKeys = Hyperparameter(
                    config, self, name + "." + key
                ).getFlatParameterNames()
                for key in subKeys:
                    keys.add(key)

            return keys
        elif self.config["type"] == "number":
            return [name]

    def getFlatParameters(self):
        name = self.root
        if "anyOf" in self.config or "oneOf" in self.config:
            parameters = []
            v = "anyOf" if "anyOf" in self.config else "oneOf"
            data = self.config[v]

            for index, param in enumerate(data):
                subParameters = Hyperparameter(
                    param, self, name + "." + str(index)
                ).getFlatParameters()
                parameters = parameters + subParameters
            return parameters
        elif "enum" in self.config or "constant" in self.config:
            return [self]
        elif self.config["type"] == "object":
            parameters = []
            for key in self.config["properties"].keys():
                config = self.config["properties"][key]
                subParameters = Hyperparameter(
                    config, self, name + "." + key
                ).getFlatParameters()
                parameters = parameters + subParameters
            return parameters
        elif self.config["type"] == "number":
            return [self]

    def getLog10Cardinality(self):
        if "anyOf" in self.config or "oneOf" in self.config:
            v = "anyOf" if "anyOf" in self.config else "oneOf"
            data = self.config[v]

            log10_cardinality = Hyperparameter(
                data[0], self, self.root + ".0"
            ).getLog10Cardinality()
            for index, subParam in enumerate(data[1:]):
                # We used logarithm identities to create this reduction formula
                other_log10_cardinality = Hyperparameter(
                    subParam, self, self.root + "." + str(index)
                ).getLog10Cardinality()

                # Revert to linear at high and low values, for numerical stability. Check here: https://www.desmos.com/calculator/efkbbftd18 to observe
                if (log10_cardinality - other_log10_cardinality) > 3:
                    log10_cardinality = log10_cardinality + 1
                elif (other_log10_cardinality - log10_cardinality) > 3:
                    log10_cardinality = other_log10_cardinality + 1
                else:
                    log10_cardinality = other_log10_cardinality + math.log10(
                        1 + math.pow(10, log10_cardinality - other_log10_cardinality)
                    )
            return log10_cardinality + math.log10(len(data))
        elif "enum" in self.config:
            return math.log10(len(self.config["enum"]))
        elif "constant" in self.config:
            return math.log10(1)
        elif self.config["type"] == "object":
            log10_cardinality = 0
            for index, subParam in enumerate(self.config["properties"].values()):
                subParameter = Hyperparameter(
                    subParam, self, self.root + "." + str(index)
                )
                log10_cardinality += subParameter.getLog10Cardinality()
            return log10_cardinality
        elif self.config["type"] == "number":
            if "rounding" in self.config:
                return math.log10(
                    min(
                        20,
                        (self.config["max"] - self.config["min"])
                        / self.config["rounding"]
                        + 1,
                    )
                )
            else:
                # Default of 20 for fully uniform numbers.
                return math.log10(20)

    def convertToFlatValues(self, params):
        flatParams = {}

        def recurse(key, value, root):
            result_key = root + "." + key
            if isinstance(value, str):
                flatParams[result_key[1:]] = value
            elif (
                isinstance(value, float)
                or isinstance(value, bool)
                or isinstance(value, int)
                or numpy.issubdtype(value, numpy.integer)
                or numpy.issubdtype(value, numpy.floating)
            ):
                flatParams[result_key[1:]] = value
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    recurse(subkey, subvalue, result_key)

        for key in params.keys():
            value = params[key]
            recurse(key, value, "")

        flatValues = {}

        if "anyOf" in self.config or "oneOf" in self.config:
            v = "anyOf" if "anyOf" in self.config else "oneOf"
            data = self.config[v]

            subParameterIndex = flatParams[self.resultVariableName + ".$index"]
            flatValues[self.name] = subParameterIndex

            for index, param in enumerate(data):
                subParameter = Hyperparameter(param, self, self.root + "." + str(index))

                if index == subParameterIndex:
                    subFlatValues = subParameter.convertToFlatValues(flatParams)
                    for key in subFlatValues:
                        flatValues[key] = subFlatValues[key]
                else:
                    for flatParam in subParameter.getFlatParameters():
                        flatValues[flatParam.name] = ""

            return flatValues
        elif "constant" in self.config:
            flatValues[self.name] = flatParams[self.resultVariableName]
            return flatValues
        elif "enum" in self.config:
            flatValues[self.name] = flatParams[self.resultVariableName]
            return flatValues
        elif self.config["type"] == "object":
            for key in self.config["properties"].keys():
                config = self.config["properties"][key]

                subFlatValues = Hyperparameter(
                    config, self, self.root + "." + key
                ).convertToFlatValues(flatParams)

                for key in subFlatValues:
                    flatValues[key] = subFlatValues[key]

                if self.name == "":
                    for key in params.keys():
                        if key.startswith("$"):
                            flatValues[key] = params[key]

            return flatValues
        elif self.config["type"] == "number":
            flatValues[self.name] = flatParams[self.resultVariableName]
            return flatValues

    def convertToStructuredValues(self, flatValues):
        if "anyOf" in self.config or "oneOf" in self.config:
            v = "anyOf" if "anyOf" in self.config else "oneOf"
            data = self.config[v]

            subParameterIndex = flatValues[self.name]
            subParam = Hyperparameter(
                data[subParameterIndex], self, self.root + "." + str(subParameterIndex)
            )

            structured = subParam.convertToStructuredValues(flatValues)
            structured["$index"] = subParameterIndex

            return structured
        elif "constant" in self.config:
            return flatValues[self.name]
        elif "enum" in self.config:
            return flatValues[self.name]
        elif self.config["type"] == "object":
            result = {}
            for key in self.config["properties"].keys():
                config = self.config["properties"][key]

                subStructuredValue = Hyperparameter(
                    config, self, self.root + "." + key
                ).convertToStructuredValues(flatValues)

                result[key] = subStructuredValue

                if self.name == "":
                    for key in flatValues.keys():
                        if key.startswith("$"):
                            result[key] = flatValues[key]
            return result
        elif self.config["type"] == "number":
            return flatValues[self.name]

    @staticmethod
    def createHyperparameterConfigForHyperoptDomain(domain):
        if domain.name is None:
            data = {"type": "object", "properties": {}}

            for key in domain.params:
                data["properties"][
                    key
                ] = Hyperparameter.createHyperparameterConfigForHyperoptDomain(
                    domain.params[key]
                )

                if "name" not in data["properties"][key]:
                    data["properties"][key]["name"] = key

            return data
        elif domain.name == "dict":
            data = {"type": "object", "properties": {}}

            for item in domain.named_args:
                data["properties"][
                    item[0]
                ] = Hyperparameter.createHyperparameterConfigForHyperoptDomain(item[1])

            return data
        elif domain.name == "switch":
            data = {"oneOf": []}
            data["name"] = domain.pos_args[0].pos_args

            for item in domain.pos_args[1:]:
                data["oneOf"].append(
                    Hyperparameter.createHyperparameterConfigForHyperoptDomain(item)
                )
            return data
        elif domain.name == "hyperopt_param":
            data = Hyperparameter.createHyperparameterConfigForHyperoptDomain(
                domain.pos_args[1]
            )
            data["name"] = domain.pos_args[0]._obj
            return data
        elif domain.name == "uniform":
            data = {"type": "number"}
            data["scaling"] = "linear"
            data["mode"] = "uniform"
            data["min"] = domain.pos_args[0]._obj
            data["max"] = domain.pos_args[1]._obj
            return data
        elif domain.name == "quniform":
            data = {"type": "number"}
            data["scaling"] = "linear"
            data["mode"] = "uniform"
            data["min"] = domain.pos_args[0]._obj
            data["max"] = domain.pos_args[1]._obj
            data["rounding"] = domain.pos_args[2]._obj
            return data
        elif domain.name == "loguniform":
            data = {"type": "number"}
            data["scaling"] = "logarithmic"
            data["mode"] = "uniform"
            data["min"] = math.exp(domain.pos_args[0]._obj)
            data["max"] = math.exp(domain.pos_args[1]._obj)
            return data
        elif domain.name == "qloguniform":
            data = {"type": "number"}
            data["scaling"] = "logarithmic"
            data["mode"] = "uniform"
            data["min"] = math.exp(domain.pos_args[0]._obj)
            data["max"] = math.exp(domain.pos_args[1]._obj)
            data["rounding"] = domain.pos_args[2]._obj
            return data
        elif domain.name == "normal":
            data = {"type": "number"}
            data["scaling"] = "linear"
            data["mode"] = "normal"
            data["mean"] = domain.pos_args[0]._obj
            data["stddev"] = domain.pos_args[1]._obj
            return data
        elif domain.name == "qnormal":
            data = {"type": "number"}
            data["scaling"] = "linear"
            data["mode"] = "normal"
            data["mean"] = domain.pos_args[0]._obj
            data["stddev"] = domain.pos_args[1]._obj
            data["rounding"] = domain.pos_args[2]._obj
            return data
        elif domain.name == "lognormal":
            data = {"type": "number"}
            data["scaling"] = "logarithmic"
            data["mode"] = "normal"
            data["mean"] = math.exp(domain.pos_args[0]._obj)
            data["stddev"] = math.exp(domain.pos_args[1]._obj)
            return data
        elif domain.name == "qlognormal":
            data = {"type": "number"}
            data["scaling"] = "logarithmic"
            data["mode"] = "normal"
            data["mean"] = math.exp(domain.pos_args[0]._obj)
            data["stddev"] = math.exp(domain.pos_args[1]._obj)
            data["rounding"] = domain.pos_args[2]._obj
            return data
        elif domain.name == "literal":
            data = {"type": "string", "constant": domain._obj}
            return data
        elif domain.name == "randint":
            data = {"type": "number"}
            low = domain.pos_args[0]._obj
            high = domain.pos_args[1]._obj if len(domain.pos_args) > 1 else None
            data["min"] = 0 if high is None else low
            data["max"] = high or low
            data["mode"] = "randint"
            return data
        else:
            raise ValueError("Unsupported hyperopt domain type " + str(domain))


class ATPEOptimizer:
    resultInformationKeys = ["trial", "status", "loss", "time", "log", "error"]

    atpeParameters = [
        "gamma",
        "nEICandidates",
        "resultFilteringAgeMultiplier",
        "resultFilteringLossRankMultiplier",
        "resultFilteringMode",
        "resultFilteringRandomProbability",
        "secondaryCorrelationExponent",
        "secondaryCorrelationMultiplier",
        "secondaryCutoff",
        "secondaryFixedProbability",
        "secondaryLockingMode",
        "secondaryProbabilityMode",
        "secondaryTopLockingPercentile",
    ]

    atpeParameterCascadeOrdering = [
        "resultFilteringMode",
        "secondaryProbabilityMode",
        "secondaryLockingMode",
        "resultFilteringAgeMultiplier",
        "resultFilteringLossRankMultiplier",
        "resultFilteringRandomProbability",
        "secondaryTopLockingPercentile",
        "secondaryCorrelationExponent",
        "secondaryCorrelationMultiplier",
        "secondaryFixedProbability",
        "secondaryCutoff",
        "gamma",
        "nEICandidates",
    ]

    atpeParameterValues = {
        "resultFilteringMode": ["age", "loss_rank", "none", "random"],
        "secondaryLockingMode": ["random", "top"],
        "secondaryProbabilityMode": ["correlation", "fixed"],
    }

    atpeModelFeatureKeys = [
        "all_correlation_best_percentile25_ratio",
        "all_correlation_best_percentile50_ratio",
        "all_correlation_best_percentile75_ratio",
        "all_correlation_kurtosis",
        "all_correlation_percentile5_percentile25_ratio",
        "all_correlation_skew",
        "all_correlation_stddev_best_ratio",
        "all_correlation_stddev_median_ratio",
        "all_loss_best_percentile25_ratio",
        "all_loss_best_percentile50_ratio",
        "all_loss_best_percentile75_ratio",
        "all_loss_kurtosis",
        "all_loss_percentile5_percentile25_ratio",
        "all_loss_skew",
        "all_loss_stddev_best_ratio",
        "all_loss_stddev_median_ratio",
        "log10_cardinality",
        "recent_10_correlation_best_percentile25_ratio",
        "recent_10_correlation_best_percentile50_ratio",
        "recent_10_correlation_best_percentile75_ratio",
        "recent_10_correlation_kurtosis",
        "recent_10_correlation_percentile5_percentile25_ratio",
        "recent_10_correlation_skew",
        "recent_10_correlation_stddev_best_ratio",
        "recent_10_correlation_stddev_median_ratio",
        "recent_10_loss_best_percentile25_ratio",
        "recent_10_loss_best_percentile50_ratio",
        "recent_10_loss_best_percentile75_ratio",
        "recent_10_loss_kurtosis",
        "recent_10_loss_percentile5_percentile25_ratio",
        "recent_10_loss_skew",
        "recent_10_loss_stddev_best_ratio",
        "recent_10_loss_stddev_median_ratio",
        "recent_15%_correlation_best_percentile25_ratio",
        "recent_15%_correlation_best_percentile50_ratio",
        "recent_15%_correlation_best_percentile75_ratio",
        "recent_15%_correlation_kurtosis",
        "recent_15%_correlation_percentile5_percentile25_ratio",
        "recent_15%_correlation_skew",
        "recent_15%_correlation_stddev_best_ratio",
        "recent_15%_correlation_stddev_median_ratio",
        "recent_15%_loss_best_percentile25_ratio",
        "recent_15%_loss_best_percentile50_ratio",
        "recent_15%_loss_best_percentile75_ratio",
        "recent_15%_loss_kurtosis",
        "recent_15%_loss_percentile5_percentile25_ratio",
        "recent_15%_loss_skew",
        "recent_15%_loss_stddev_best_ratio",
        "recent_15%_loss_stddev_median_ratio",
        "recent_25_correlation_best_percentile25_ratio",
        "recent_25_correlation_best_percentile50_ratio",
        "recent_25_correlation_best_percentile75_ratio",
        "recent_25_correlation_kurtosis",
        "recent_25_correlation_percentile5_percentile25_ratio",
        "recent_25_correlation_skew",
        "recent_25_correlation_stddev_best_ratio",
        "recent_25_correlation_stddev_median_ratio",
        "recent_25_loss_best_percentile25_ratio",
        "recent_25_loss_best_percentile50_ratio",
        "recent_25_loss_best_percentile75_ratio",
        "recent_25_loss_kurtosis",
        "recent_25_loss_percentile5_percentile25_ratio",
        "recent_25_loss_skew",
        "recent_25_loss_stddev_best_ratio",
        "recent_25_loss_stddev_median_ratio",
        "top_10%_correlation_best_percentile25_ratio",
        "top_10%_correlation_best_percentile50_ratio",
        "top_10%_correlation_best_percentile75_ratio",
        "top_10%_correlation_kurtosis",
        "top_10%_correlation_percentile5_percentile25_ratio",
        "top_10%_correlation_skew",
        "top_10%_correlation_stddev_best_ratio",
        "top_10%_correlation_stddev_median_ratio",
        "top_10%_loss_best_percentile25_ratio",
        "top_10%_loss_best_percentile50_ratio",
        "top_10%_loss_best_percentile75_ratio",
        "top_10%_loss_kurtosis",
        "top_10%_loss_percentile5_percentile25_ratio",
        "top_10%_loss_skew",
        "top_10%_loss_stddev_best_ratio",
        "top_10%_loss_stddev_median_ratio",
        "top_20%_correlation_best_percentile25_ratio",
        "top_20%_correlation_best_percentile50_ratio",
        "top_20%_correlation_best_percentile75_ratio",
        "top_20%_correlation_kurtosis",
        "top_20%_correlation_percentile5_percentile25_ratio",
        "top_20%_correlation_skew",
        "top_20%_correlation_stddev_best_ratio",
        "top_20%_correlation_stddev_median_ratio",
        "top_20%_loss_best_percentile25_ratio",
        "top_20%_loss_best_percentile50_ratio",
        "top_20%_loss_best_percentile75_ratio",
        "top_20%_loss_kurtosis",
        "top_20%_loss_percentile5_percentile25_ratio",
        "top_20%_loss_skew",
        "top_20%_loss_stddev_best_ratio",
        "top_20%_loss_stddev_median_ratio",
        "top_30%_correlation_best_percentile25_ratio",
        "top_30%_correlation_best_percentile50_ratio",
        "top_30%_correlation_best_percentile75_ratio",
        "top_30%_correlation_kurtosis",
        "top_30%_correlation_percentile5_percentile25_ratio",
        "top_30%_correlation_skew",
        "top_30%_correlation_stddev_best_ratio",
        "top_30%_correlation_stddev_median_ratio",
        "top_30%_loss_best_percentile25_ratio",
        "top_30%_loss_best_percentile50_ratio",
        "top_30%_loss_best_percentile75_ratio",
        "top_30%_loss_kurtosis",
        "top_30%_loss_percentile5_percentile25_ratio",
        "top_30%_loss_skew",
        "top_30%_loss_stddev_best_ratio",
        "top_30%_loss_stddev_median_ratio",
    ]

    def __init__(self):
        try:
            import lightgbm
            import sklearn
        except ImportError:
            raise ImportError(
                "You must install lightgbm and sklearn in order to use the ATPE algorithm. Please run `pip install lightgbm scikit-learn` and try again. These are not built in dependencies of hyperopt."
            )

        scalingModelData = json.loads(
            pkg_resources.resource_string(
                __name__, "atpe_models/scaling_model.json"
            ).decode("utf-8")
        )
        self.featureScalingModels = {}
        for key in self.atpeModelFeatureKeys:
            self.featureScalingModels[key] = sklearn.preprocessing.StandardScaler()
            self.featureScalingModels[key].scale_ = numpy.array(
                scalingModelData[key]["scales"]
            )
            self.featureScalingModels[key].mean_ = numpy.array(
                scalingModelData[key]["means"]
            )
            self.featureScalingModels[key].var_ = numpy.array(
                scalingModelData[key]["variances"]
            )
            self.featureScalingModels[key].n_features_in_ = 1

        self.parameterModels = {}
        self.parameterModelConfigurations = {}
        for param in self.atpeParameters:
            modelData = pkg_resources.resource_string(
                __name__, "atpe_models/model-" + param + ".txt"
            )
            with ClosedNamedTempFile(modelData) as model_file_name:
                self.parameterModels[param] = lightgbm.Booster(
                    model_file=model_file_name
                )

            configString = pkg_resources.resource_string(
                __name__, "atpe_models/model-" + param + "-configuration.json"
            )
            data = json.loads(configString.decode("utf-8"))
            self.parameterModelConfigurations[param] = data

        self.lastATPEParameters = None
        self.lastLockedParameters = []
        self.atpeParamDetails = None

    def recommendNextParameters(
        self, hyperparameterSpace, results, currentTrials, lockedValues=None
    ):
        rstate = numpy.random.default_rng(seed=int(random.randint(1, 2 ** 32 - 1)))

        params = {"param": {}}

        def sample(parameters):
            params["param"] = parameters
            return {"loss": 0.5, "status": "ok"}

        parameters = Hyperparameter(hyperparameterSpace).getFlatParameters()

        if lockedValues is not None:
            # Remove any locked values from ones the optimizer will examine
            parameters = list(
                filter(lambda param: param.name not in lockedValues.keys(), parameters)
            )

        log10_cardinality = Hyperparameter(hyperparameterSpace).getLog10Cardinality()
        initializationRounds = max(10, int(log10_cardinality))

        atpeParams = {}
        atpeParamDetails = {}
        if (
            len(list(result for result in results if result["loss"]))
            < initializationRounds
        ):
            atpeParams = {
                "gamma": 1.0,
                "nEICandidates": 24,
                "resultFilteringAgeMultiplier": None,
                "resultFilteringLossRankMultiplier": None,
                "resultFilteringMode": "none",
                "resultFilteringRandomProbability": None,
                "secondaryCorrelationExponent": 1.0,
                "secondaryCorrelationMultiplier": None,
                "secondaryCutoff": 0,
                "secondarySorting": 0,
                "secondaryFixedProbability": 0.5,
                "secondaryLockingMode": "top",
                "secondaryProbabilityMode": "fixed",
                "secondaryTopLockingPercentile": 0,
            }
        else:
            # Calculate the statistics for the distribution
            stats = self.computeAllResultStatistics(hyperparameterSpace, results)
            stats["num_parameters"] = len(parameters)
            stats["log10_cardinality"] = Hyperparameter(
                hyperparameterSpace
            ).getLog10Cardinality()
            stats["log10_trial"] = math.log10(len(results))
            baseVector = []

            for feature in self.atpeModelFeatureKeys:
                scalingModel = self.featureScalingModels[feature]
                transformed = scalingModel.transform([[stats[feature]]])[0][0]
                baseVector.append(transformed)

            baseVector = numpy.array([baseVector])

            for atpeParamIndex, atpeParameter in enumerate(
                self.atpeParameterCascadeOrdering
            ):
                vector = copy.copy(baseVector)[0].tolist()
                atpeParamFeatures = self.atpeParameterCascadeOrdering[:atpeParamIndex]
                for atpeParamFeature in atpeParamFeatures:
                    # We have to insert a special value of -3 for any conditional parameters.
                    if (
                        atpeParamFeature == "resultFilteringAgeMultiplier"
                        and atpeParams["resultFilteringMode"] != "age"
                    ):
                        vector.append(
                            -3
                        )  # This is the default value inserted when parameters aren't relevant
                    elif (
                        atpeParamFeature == "resultFilteringLossRankMultiplier"
                        and atpeParams["resultFilteringMode"] != "loss_rank"
                    ):
                        vector.append(
                            -3
                        )  # This is the default value inserted when parameters aren't relevant
                    elif (
                        atpeParamFeature == "resultFilteringRandomProbability"
                        and atpeParams["resultFilteringMode"] != "random"
                    ):
                        vector.append(
                            -3
                        )  # This is the default value inserted when parameters aren't relevant
                    elif (
                        atpeParamFeature == "secondaryCorrelationMultiplier"
                        and atpeParams["secondaryProbabilityMode"] != "correlation"
                    ):
                        vector.append(
                            -3
                        )  # This is the default value inserted when parameters aren't relevant
                    elif (
                        atpeParamFeature == "secondaryFixedProbability"
                        and atpeParams["secondaryProbabilityMode"] != "fixed"
                    ):
                        vector.append(
                            -3
                        )  # This is the default value inserted when parameters aren't relevant
                    elif (
                        atpeParamFeature == "secondaryTopLockingPercentile"
                        and atpeParams["secondaryLockingMode"] != "top"
                    ):
                        vector.append(
                            -3
                        )  # This is the default value inserted when parameters aren't relevant
                    elif atpeParamFeature in self.atpeParameterValues:
                        for value in self.atpeParameterValues[atpeParamFeature]:
                            vector.append(
                                1.0 if atpeParams[atpeParamFeature] == value else 0
                            )
                    else:
                        vector.append(float(atpeParams[atpeParamFeature]))

                allFeatureKeysForATPEParamModel = copy.copy(self.atpeModelFeatureKeys)
                for atpeParamFeature in atpeParamFeatures:
                    if atpeParamFeature in self.atpeParameterValues:
                        for value in self.atpeParameterValues[atpeParamFeature]:
                            allFeatureKeysForATPEParamModel.append(
                                atpeParamFeature + "_" + value
                            )
                    else:
                        allFeatureKeysForATPEParamModel.append(atpeParamFeature)

                value = self.parameterModels[atpeParameter].predict([vector])[0]
                featureContributions = self.parameterModels[atpeParameter].predict(
                    [vector], pred_contrib=True
                )[0]

                atpeParamDetails[atpeParameter] = {"value": None, "reason": None}

                # Set the value
                if atpeParameter in self.atpeParameterValues:
                    # Renormalize the predicted probabilities
                    config = self.parameterModelConfigurations[atpeParameter]
                    for atpeParamValueIndex, atpeParamValue in enumerate(
                        self.atpeParameterValues[atpeParameter]
                    ):
                        value[atpeParamValueIndex] = (
                            (
                                (
                                    value[atpeParamValueIndex]
                                    - config["predMeans"][atpeParamValue]
                                )
                                / config["predStddevs"][atpeParamValue]
                            )
                            * config["origStddevs"][atpeParamValue]
                        ) + config["origMeans"][atpeParamValue]
                        value[atpeParamValueIndex] = max(
                            0.0, min(1.0, value[atpeParamValueIndex])
                        )

                    maxVal = numpy.max(value)
                    for atpeParamValueIndex, atpeParamValue in enumerate(
                        self.atpeParameterValues[atpeParameter]
                    ):
                        value[atpeParamValueIndex] = max(
                            value[atpeParamValueIndex], maxVal * 0.15
                        )  # We still allow the non recommended modes to get chosen 15% of the time

                    # Make a random weighted choice based on the normalized probabilities
                    probabilities = value / numpy.sum(value)
                    chosen = numpy.random.choice(
                        a=self.atpeParameterValues[atpeParameter], p=probabilities
                    )
                    atpeParams[atpeParameter] = str(chosen)
                else:
                    # Renormalize the predictions
                    config = self.parameterModelConfigurations[atpeParameter]
                    value = (
                        ((value - config["predMean"]) / config["predStddev"])
                        * config["origStddev"]
                    ) + config["origMean"]
                    atpeParams[atpeParameter] = float(value)

                atpeParamDetails[atpeParameter]["reason"] = {}
                # If we are predicting a class, we get separate feature contributions for each class. Take the average
                if atpeParameter in self.atpeParameterValues:
                    featureContributions = numpy.mean(
                        numpy.reshape(
                            featureContributions,
                            newshape=(
                                len(allFeatureKeysForATPEParamModel) + 1,
                                len(self.atpeParameterValues[atpeParameter]),
                            ),
                        ),
                        axis=1,
                    )

                contributions = [
                    (
                        self.atpeModelFeatureKeys[index],
                        float(featureContributions[index]),
                    )
                    for index in range(len(self.atpeModelFeatureKeys))
                ]
                contributions = sorted(contributions, key=lambda r: -r[1])
                # Only focus on the top 10% of features, since it gives more useful information. Otherwise the total gets really squashed out over many features,
                # because our model is highly regularized.
                contributions = contributions[: int(len(contributions) / 10)]
                total = numpy.sum([contrib[1] for contrib in contributions])

                for contributionIndex, contribution in enumerate(contributions[:3]):
                    atpeParamDetails[atpeParameter]["reason"][contribution[0]] = (
                        str(int(float(contribution[1]) * 100.0 / total)) + "%"
                    )

                # Apply bounds to all the parameters
                if atpeParameter == "gamma":
                    atpeParams["gamma"] = max(0.2, min(2.0, atpeParams["gamma"]))
                if atpeParameter == "nEICandidates":
                    atpeParams["nEICandidates"] = int(
                        max(2.0, min(48, atpeParams["nEICandidates"]))
                    )
                if atpeParameter == "resultFilteringAgeMultiplier":
                    atpeParams["resultFilteringAgeMultiplier"] = max(
                        1.0, min(4.0, atpeParams["resultFilteringAgeMultiplier"])
                    )
                if atpeParameter == "resultFilteringLossRankMultiplier":
                    atpeParams["resultFilteringLossRankMultiplier"] = max(
                        1.0, min(4.0, atpeParams["resultFilteringLossRankMultiplier"])
                    )
                if atpeParameter == "resultFilteringRandomProbability":
                    atpeParams["resultFilteringRandomProbability"] = max(
                        0.7, min(0.9, atpeParams["resultFilteringRandomProbability"])
                    )
                if atpeParameter == "secondaryCorrelationExponent":
                    atpeParams["secondaryCorrelationExponent"] = max(
                        1.0, min(3.0, atpeParams["secondaryCorrelationExponent"])
                    )
                if atpeParameter == "secondaryCorrelationMultiplier":
                    atpeParams["secondaryCorrelationMultiplier"] = max(
                        0.2, min(1.8, atpeParams["secondaryCorrelationMultiplier"])
                    )
                if atpeParameter == "secondaryCutoff":
                    atpeParams["secondaryCutoff"] = max(
                        -1.0, min(1.0, atpeParams["secondaryCutoff"])
                    )
                if atpeParameter == "secondaryFixedProbability":
                    atpeParams["secondaryFixedProbability"] = max(
                        0.2, min(0.8, atpeParams["secondaryFixedProbability"])
                    )
                if atpeParameter == "secondaryTopLockingPercentile":
                    atpeParams["secondaryTopLockingPercentile"] = max(
                        0, min(10.0, atpeParams["secondaryTopLockingPercentile"])
                    )

            # Now blank out unneeded params so they don't confuse us
            if atpeParams["secondaryLockingMode"] == "random":
                atpeParams["secondaryTopLockingPercentile"] = None

            if atpeParams["secondaryProbabilityMode"] == "fixed":
                atpeParams["secondaryCorrelationMultiplier"] = None
            else:
                atpeParams["secondaryFixedProbability"] = None

            if atpeParams["resultFilteringMode"] == "none":
                atpeParams["resultFilteringAgeMultiplier"] = None
                atpeParams["resultFilteringLossRankMultiplier"] = None
                atpeParams["resultFilteringRandomProbability"] = None
            elif atpeParams["resultFilteringMode"] == "age":
                atpeParams["resultFilteringLossRankMultiplier"] = None
                atpeParams["resultFilteringRandomProbability"] = None
            elif atpeParams["resultFilteringMode"] == "loss_rank":
                atpeParams["resultFilteringAgeMultiplier"] = None
                atpeParams["resultFilteringRandomProbability"] = None
            elif atpeParams["resultFilteringMode"] == "random":
                atpeParams["resultFilteringAgeMultiplier"] = None
                atpeParams["resultFilteringLossRankMultiplier"] = None

            for atpeParameter in self.atpeParameters:
                if atpeParams[atpeParameter] is None:
                    del atpeParamDetails[atpeParameter]
                else:
                    atpeParamDetails[atpeParameter]["value"] = atpeParams[atpeParameter]

        self.lastATPEParameters = atpeParams
        self.atpeParamDetails = atpeParamDetails

        def computePrimarySecondary():
            if len(results) < initializationRounds:
                return (
                    parameters,
                    [],
                    [0.5] * len(parameters),
                )  # Put all parameters as primary

            if len({result["loss"] for result in results}) < 5:
                return (
                    parameters,
                    [],
                    [0.5] * len(parameters),
                )  # Put all parameters as primary

            numberParameters = [
                parameter
                for parameter in parameters
                if parameter.config["type"] == "number"
            ]
            otherParameters = [
                parameter
                for parameter in parameters
                if parameter.config["type"] != "number"
            ]

            totalWeight = 0
            correlations = {}
            for parameter in numberParameters:
                if (
                    len(
                        {
                            result[parameter.name]
                            for result in results
                            if result[parameter.name] is not None
                        }
                    )
                    < 2
                ):
                    correlations[parameter.name] = 0
                else:
                    values = []
                    valueLosses = []
                    for result in results:
                        if (
                            result[parameter.name] is not None
                            and result["loss"] is not None
                        ):
                            values.append(result[parameter.name])
                            valueLosses.append(result["loss"])

                    correlation = math.pow(
                        abs(scipy.stats.spearmanr(values, valueLosses)[0]),
                        atpeParams["secondaryCorrelationExponent"],
                    )
                    correlations[parameter.name] = correlation
                    totalWeight += correlation

            threshold = totalWeight * abs(atpeParams["secondaryCutoff"])

            if atpeParams["secondaryCutoff"] < 0:
                # Reverse order - we lock in the highest correlated parameters
                sortedParameters = sorted(
                    numberParameters, key=lambda parameter: correlations[parameter.name]
                )
            else:
                # Normal order - sort properties by their correlation to lock in lowest correlated parameters
                sortedParameters = sorted(
                    numberParameters,
                    key=lambda parameter: -correlations[parameter.name],
                )

            primaryParameters = []
            secondaryParameters = []
            cumulative = totalWeight
            for parameter in sortedParameters:
                if cumulative < threshold:
                    secondaryParameters.append(parameter)
                else:
                    primaryParameters.append(parameter)

                cumulative -= correlations[parameter.name]

            return (
                primaryParameters + otherParameters,
                secondaryParameters,
                correlations,
            )

        if (
            len([result["loss"] for result in results if result["loss"] is not None])
            == 0
        ):
            maxLoss = 1
        else:
            maxLoss = numpy.max(
                [result["loss"] for result in results if result["loss"] is not None]
            )

        # We create a copy of lockedValues so we don't modify the object that was passed in as an argument - treat it as immutable.
        # The ATPE algorithm will lock additional values in a stochastic manner
        if lockedValues is None:
            lockedValues = {}
        else:
            lockedValues = copy.copy(lockedValues)

        filteredResults = []
        removedResults = []
        if len(results) > initializationRounds:
            (
                primaryParameters,
                secondaryParameters,
                correlations,
            ) = computePrimarySecondary()

            self.lastLockedParameters = []

            sortedResults = list(
                sorted(
                    list(results),
                    key=lambda result: (
                        result["loss"] if result["loss"] is not None else (maxLoss + 1)
                    ),
                )
            )
            topResults = sortedResults
            if atpeParams["secondaryLockingMode"] == "top":
                topResultsN = max(
                    1,
                    int(
                        math.ceil(
                            len(sortedResults)
                            * atpeParams["secondaryTopLockingPercentile"]
                            / 100.0
                        )
                    ),
                )
                topResults = sortedResults[:topResultsN]

            # Any secondary parameters have may be locked to either the current best
            # value or any value within the result pool.
            for secondary in secondaryParameters:
                if atpeParams["secondaryProbabilityMode"] == "fixed":
                    if random.uniform(0, 1) < atpeParams["secondaryFixedProbability"]:
                        self.lastLockedParameters.append(secondary.name)
                        if atpeParams["secondaryLockingMode"] == "top":
                            lockResult = random.choice(topResults)
                            if (
                                lockResult[secondary.name] is not None
                                and lockResult[secondary.name] != ""
                            ):
                                lockedValues[secondary.name] = lockResult[
                                    secondary.name
                                ]
                        elif atpeParams["secondaryLockingMode"] == "random":
                            lockedValues[
                                secondary.name
                            ] = self.chooseRandomValueForParameter(secondary)

                elif atpeParams["secondaryProbabilityMode"] == "correlation":
                    probability = max(
                        0,
                        min(
                            1,
                            abs(correlations[secondary.name])
                            * atpeParams["secondaryCorrelationMultiplier"],
                        ),
                    )
                    if random.uniform(0, 1) < probability:
                        self.lastLockedParameters.append(secondary.name)
                        if atpeParams["secondaryLockingMode"] == "top":
                            lockResult = random.choice(topResults)
                            if (
                                lockResult[secondary.name] is not None
                                and lockResult[secondary.name] != ""
                            ):
                                lockedValues[secondary.name] = lockResult[
                                    secondary.name
                                ]
                        elif atpeParams["secondaryLockingMode"] == "random":
                            lockedValues[
                                secondary.name
                            ] = self.chooseRandomValueForParameter(secondary)

            # Now last step, we filter results prior to sending them into ATPE
            for resultIndex, result in enumerate(results):
                if atpeParams["resultFilteringMode"] == "none":
                    filteredResults.append(result)
                elif atpeParams["resultFilteringMode"] == "random":
                    if (
                        random.uniform(0, 1)
                        < atpeParams["resultFilteringRandomProbability"]
                    ):
                        filteredResults.append(result)
                    else:
                        removedResults.append(result)
                elif atpeParams["resultFilteringMode"] == "age":
                    age = float(resultIndex) / float(len(results))
                    if random.uniform(0, 1) < (
                        atpeParams["resultFilteringAgeMultiplier"] * age
                    ):
                        filteredResults.append(result)
                    else:
                        removedResults.append(result)
                elif atpeParams["resultFilteringMode"] == "loss_rank":
                    rank = 1.0 - (
                        float(sortedResults.index(result)) / float(len(results))
                    )
                    if random.uniform(0, 1) < (
                        atpeParams["resultFilteringLossRankMultiplier"] * rank
                    ):
                        filteredResults.append(result)
                    else:
                        removedResults.append(result)

        # If we are in initialization, or by some other fluke of random nature that we
        # end up with no results after filtering, then just use all the results
        if len(filteredResults) == 0:
            filteredResults = results

        hyperopt.fmin(
            fn=sample,
            space=Hyperparameter(hyperparameterSpace).createHyperoptSpace(lockedValues),
            algo=functools.partial(
                hyperopt.tpe.suggest,
                n_startup_jobs=initializationRounds,
                gamma=atpeParams["gamma"],
                n_EI_candidates=int(atpeParams["nEICandidates"]),
            ),
            max_evals=1,
            trials=self.convertResultsToTrials(hyperparameterSpace, filteredResults),
            rstate=rstate,
            show_progressbar=False,
        )

        return params.get("param")

    def chooseRandomValueForParameter(self, parameter):
        if parameter.config.get("mode", "uniform") == "uniform":
            minVal = parameter.config["min"]
            maxVal = parameter.config["max"]

            if parameter.config.get("scaling", "linear") == "logarithmic":
                minVal = math.log(minVal)
                maxVal = math.log(maxVal)

            value = random.uniform(minVal, maxVal)

            if parameter.config.get("scaling", "linear") == "logarithmic":
                value = math.exp(value)

            if "rounding" in parameter.config:
                value = (
                    round(value / parameter.config["rounding"])
                    * parameter.config["rounding"]
                )
        elif parameter.config.get("mode", "uniform") == "normal":
            meanVal = parameter.config["mean"]
            stddevVal = parameter.config["stddev"]

            if parameter.config.get("scaling", "linear") == "logarithmic":
                meanVal = math.log(meanVal)
                stddevVal = math.log(stddevVal)

            value = random.gauss(meanVal, stddevVal)

            if parameter.config.get("scaling", "linear") == "logarithmic":
                value = math.exp(value)

            if "rounding" in parameter.config:
                value = (
                    round(value / parameter.config["rounding"])
                    * parameter.config["rounding"]
                )
        elif parameter.config.get("mode", "uniform") == "randint":
            min = parameter.config["min"]
            max = parameter.config["max"]
            value = random.randint(min, max)

        return value

    def computePartialResultStatistics(self, hyperparameterSpace, results):
        losses = numpy.array(
            sorted([result["loss"] for result in results if result["loss"] is not None])
        )

        bestLoss = 0
        percentile5Loss = 0
        percentile25Loss = 0
        percentile50Loss = 0
        percentile75Loss = 0
        statistics = {}

        numpy.warnings.filterwarnings("ignore")

        if len(set(losses)) > 1:
            bestLoss = numpy.percentile(losses, 0)
            percentile5Loss = numpy.percentile(losses, 5)
            percentile25Loss = numpy.percentile(losses, 25)
            percentile50Loss = numpy.percentile(losses, 50)
            percentile75Loss = numpy.percentile(losses, 75)

            statistics["loss_skew"] = scipy.stats.skew(losses)
            statistics["loss_kurtosis"] = scipy.stats.kurtosis(losses)
        else:
            statistics["loss_skew"] = 0
            statistics["loss_kurtosis"] = 0

        if percentile50Loss == 0:
            statistics["loss_stddev_median_ratio"] = 0
            statistics["loss_best_percentile50_ratio"] = 0
        else:
            statistics["loss_stddev_median_ratio"] = (
                numpy.std(losses) / percentile50Loss
            )
            statistics["loss_best_percentile50_ratio"] = bestLoss / percentile50Loss

        if bestLoss == 0:
            statistics["loss_stddev_best_ratio"] = 0
        else:
            statistics["loss_stddev_best_ratio"] = numpy.std(losses) / bestLoss

        if percentile25Loss == 0:
            statistics["loss_best_percentile25_ratio"] = 0
            statistics["loss_percentile5_percentile25_ratio"] = 0
        else:
            statistics["loss_best_percentile25_ratio"] = bestLoss / percentile25Loss
            statistics["loss_percentile5_percentile25_ratio"] = (
                percentile5Loss / percentile25Loss
            )

        if percentile75Loss == 0:
            statistics["loss_best_percentile75_ratio"] = 0
        else:
            statistics["loss_best_percentile75_ratio"] = bestLoss / percentile75Loss

        def getValue(result, parameter):
            return result[parameter.name]

        # Now we compute correlations between each parameter and the loss
        parameters = Hyperparameter(hyperparameterSpace).getFlatParameters()
        correlations = []
        for parameter in parameters:
            if parameter.config["type"] == "number":
                if (
                    len(
                        {
                            getValue(result, parameter)
                            for result in results
                            if (
                                getValue(result, parameter) is not None
                                and result["loss"] is not None
                            )
                        }
                    )
                    < 2
                ):
                    correlations.append(0)
                else:
                    values = []
                    valueLosses = []
                    for result in results:
                        if result["loss"] is not None and (
                            isinstance(getValue(result, parameter), float)
                            or isinstance(getValue(result, parameter), int)
                        ):
                            values.append(getValue(result, parameter))
                            valueLosses.append(result["loss"])

                    correlation = abs(scipy.stats.spearmanr(values, valueLosses)[0])
                    if math.isnan(correlation) or math.isinf(correlation):
                        correlations.append(0)
                    else:
                        correlations.append(correlation)

        correlations = numpy.array(correlations)

        if len(set(correlations)) == 1:
            statistics["correlation_skew"] = 0
            statistics["correlation_kurtosis"] = 0
            statistics["correlation_stddev_median_ratio"] = 0
            statistics["correlation_stddev_best_ratio"] = 0

            statistics["correlation_best_percentile25_ratio"] = 0
            statistics["correlation_best_percentile50_ratio"] = 0
            statistics["correlation_best_percentile75_ratio"] = 0
            statistics["correlation_percentile5_percentile25_ratio"] = 0
        else:
            bestCorrelation = numpy.percentile(
                correlations, 100
            )  # Correlations are in the opposite order of losses, higher correlation is considered "best"
            percentile5Correlation = numpy.percentile(correlations, 95)
            percentile25Correlation = numpy.percentile(correlations, 75)
            percentile50Correlation = numpy.percentile(correlations, 50)
            percentile75Correlation = numpy.percentile(correlations, 25)

            statistics["correlation_skew"] = scipy.stats.skew(correlations)
            statistics["correlation_kurtosis"] = scipy.stats.kurtosis(correlations)

            if percentile50Correlation == 0:
                statistics["correlation_stddev_median_ratio"] = 0
                statistics["correlation_best_percentile50_ratio"] = 0
            else:
                statistics["correlation_stddev_median_ratio"] = (
                    numpy.std(correlations) / percentile50Correlation
                )
                statistics["correlation_best_percentile50_ratio"] = (
                    bestCorrelation / percentile50Correlation
                )

            if bestCorrelation == 0:
                statistics["correlation_stddev_best_ratio"] = 0
            else:
                statistics["correlation_stddev_best_ratio"] = (
                    numpy.std(correlations) / bestCorrelation
                )

            if percentile25Correlation == 0:
                statistics["correlation_best_percentile25_ratio"] = 0
                statistics["correlation_percentile5_percentile25_ratio"] = 0
            else:
                statistics["correlation_best_percentile25_ratio"] = (
                    bestCorrelation / percentile25Correlation
                )
                statistics["correlation_percentile5_percentile25_ratio"] = (
                    percentile5Correlation / percentile25Correlation
                )

            if percentile75Correlation == 0:
                statistics["correlation_best_percentile75_ratio"] = 0
            else:
                statistics["correlation_best_percentile75_ratio"] = (
                    bestCorrelation / percentile75Correlation
                )

        return statistics

    def computeAllResultStatistics(self, hyperparameterSpace, results):
        losses = numpy.array(
            sorted([result["loss"] for result in results if result["loss"] is not None])
        )

        if len(set(losses)) > 1:
            percentile10Loss = numpy.percentile(losses, 10)
            percentile20Loss = numpy.percentile(losses, 20)
            percentile30Loss = numpy.percentile(losses, 30)
        else:
            percentile10Loss = losses[0]
            percentile20Loss = losses[0]
            percentile30Loss = losses[0]

        allResults = list(results)
        percentile10Results = [
            result
            for result in results
            if result["loss"] is not None and result["loss"] <= percentile10Loss
        ]
        percentile20Results = [
            result
            for result in results
            if result["loss"] is not None and result["loss"] <= percentile20Loss
        ]
        percentile30Results = [
            result
            for result in results
            if result["loss"] is not None and result["loss"] <= percentile30Loss
        ]

        recent10Count = min(len(results), 10)
        recent10Results = results[-recent10Count:]

        recent25Count = min(len(results), 25)
        recent25Results = results[-recent25Count:]

        recent15PercentCount = max(math.ceil(len(results) * 0.15), 5)
        recent15PercentResults = results[-recent15PercentCount:]

        statistics = {}
        allResultStatistics = self.computePartialResultStatistics(
            hyperparameterSpace, allResults
        )
        for stat, value in allResultStatistics.items():
            statistics["all_" + stat] = value

        percentile10Statistics = self.computePartialResultStatistics(
            hyperparameterSpace, percentile10Results
        )
        for stat, value in percentile10Statistics.items():
            statistics["top_10%_" + stat] = value

        percentile20Statistics = self.computePartialResultStatistics(
            hyperparameterSpace, percentile20Results
        )
        for stat, value in percentile20Statistics.items():
            statistics["top_20%_" + stat] = value

        percentile30Statistics = self.computePartialResultStatistics(
            hyperparameterSpace, percentile30Results
        )
        for stat, value in percentile30Statistics.items():
            statistics["top_30%_" + stat] = value

        recent10Statistics = self.computePartialResultStatistics(
            hyperparameterSpace, recent10Results
        )
        for stat, value in recent10Statistics.items():
            statistics["recent_10_" + stat] = value

        recent25Statistics = self.computePartialResultStatistics(
            hyperparameterSpace, recent25Results
        )
        for stat, value in recent25Statistics.items():
            statistics["recent_25_" + stat] = value

        recent15PercentResult = self.computePartialResultStatistics(
            hyperparameterSpace, recent15PercentResults
        )
        for stat, value in recent15PercentResult.items():
            statistics["recent_15%_" + stat] = value

        # Although we have added lots of protection in the computePartialResultStatistics code, one last hedge against any NaN or infinity values coming up
        # in our statistics
        for key in statistics.keys():
            if math.isnan(statistics[key]) or math.isinf(statistics[key]):
                statistics[key] = 0

        return statistics

    def convertResultsToTrials(self, hyperparameterSpace, results):
        trials = hyperopt.Trials()

        for resultIndex, result in enumerate(results):
            data = {
                "book_time": datetime.datetime.now(),
                "exp_key": None,
                "misc": {
                    "cmd": ("domain_attachment", "FMinIter_Domain"),
                    "idxs": {},
                    "tid": resultIndex,
                    "vals": {},
                    "workdir": None,
                },
                "owner": None,
                "refresh_time": datetime.datetime.now(),
                "result": {"loss": result["loss"], "status": result["status"]},
                "spec": None,
                "state": 2,
                "tid": resultIndex,
                "version": 0,
            }

            for param in Hyperparameter(hyperparameterSpace).getFlatParameters():
                value = result[param.name]
                if value not in ("", None):
                    if "enum" in param.config:
                        value = param.config["enum"].index(value)

                    data["misc"]["idxs"][param.hyperoptVariableName] = [resultIndex]
                    data["misc"]["vals"][param.hyperoptVariableName] = [value]
                else:
                    data["misc"]["idxs"][param.hyperoptVariableName] = []
                    data["misc"]["vals"][param.hyperoptVariableName] = []

            trials.insert_trial_doc(data)
        return trials

    def convertTrialsToResults(self, hyperparameterSpace, trials):
        results = []
        for trialIndex, trial in enumerate(trials.trials):
            data = {
                "trial": trialIndex,
                "status": trial["result"]["status"],
                "loss": trial["result"]["loss"],
                "log": "",
                "time": abs(
                    (trial["book_time"] - trial["refresh_time"]).total_seconds()
                ),
            }

            params = trial["misc"]["vals"]
            for param in Hyperparameter(hyperparameterSpace).getFlatParameters():
                key = param.hyperoptVariableName

                if len(params[key]) == 1:
                    value = params[key][0]
                    if "enum" in param.config:
                        value = param.config["enum"][value]

                    data[param.name] = value
                else:
                    data[param.name] = ""

            results.append(data)
        return results


def suggest(new_ids, domain, trials, seed):
    optimizer = ATPEOptimizer()

    # Convert the PyLL domain back into a descriptive form of hyperparameter space
    hyperparameterConfig = Hyperparameter.createHyperparameterConfigForHyperoptDomain(
        domain
    )

    results = optimizer.convertTrialsToResults(hyperparameterConfig, trials)

    # If there is a loss value that is negative, then we must increment the values so
    # they are all positive. This is because ATPE has been optimized only for positive
    # loss value
    if len(results) > 0:
        minVal = min(
            [result["loss"] for result in results if result["loss"] is not None]
        )
        if minVal < 0:
            for result in results:
                if result["loss"] is not None:
                    result["loss"] = result["loss"] - minVal + 0.1

    hyperparameters = Hyperparameter(hyperparameterConfig)

    rval = []
    for new_id in new_ids:
        parameters = optimizer.recommendNextParameters(
            hyperparameterConfig, results, currentTrials=[]
        )
        flatParameters = hyperparameters.convertToFlatValues(parameters)

        rval_results = [domain.new_result()]
        rval_miscs = [
            dict(
                tid=new_id,
                cmd=domain.cmd,
                workdir=domain.workdir,
                idxs={key: [0] for key in flatParameters},
                vals={key: [flatParameters[key]] for key in flatParameters},
            )
        ]

        rval.extend(trials.new_trial_docs([new_id], [None], rval_results, rval_miscs))

    return rval
