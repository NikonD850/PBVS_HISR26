import importlib
import pkgutil
import re


ARCHS_PACKAGE = "model.archs"


def _sanitize_arch_name(name: str):
    if not isinstance(name, str):
        raise ValueError("model_file 必须是字符串")
    trimmed = name.strip()
    if trimmed == "":
        raise ValueError("model_file 不能为空")
    if trimmed.endswith(".py"):
        trimmed = trimmed[:-3]
    if trimmed.startswith("model.archs."):
        trimmed = trimmed[len("model.archs.") :]
    elif trimmed.startswith("archs."):
        trimmed = trimmed[len("archs.") :]
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", trimmed):
        raise ValueError(f"非法模型名称: {name}")
    return trimmed


def list_available_archs():
    package = importlib.import_module(ARCHS_PACKAGE)
    return sorted(
        module.name
        for module in pkgutil.iter_modules(package.__path__)
        if module.name != "__init__" and not module.name.startswith("_")
    )


def _load_arch_module(arch_name: str):
    module_path = f"{ARCHS_PACKAGE}.{arch_name}"
    return importlib.import_module(module_path)


def build_model(options, device, gpu_count):
    arch_name = _sanitize_arch_name(options.model.model_file)
    try:
        module = _load_arch_module(arch_name)
    except ModuleNotFoundError as exc:
        available = ", ".join(list_available_archs()) or "<empty>"
        raise ModuleNotFoundError(
            f"未找到模型架构: {arch_name}。可用架构: {available}"
        ) from exc

    if not hasattr(module, "build_model"):
        raise AttributeError(
            f"{ARCHS_PACKAGE}.{arch_name} 缺少 build_model(options, device, gpu_count)"
        )
    return module.build_model(options=options, device=device, gpu_count=gpu_count)
