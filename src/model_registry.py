import argparse
import os

import mlflow
from mlflow.tracking import MlflowClient


def _client(tracking_uri: str, registry_uri: str) -> MlflowClient:
    mlflow.set_tracking_uri(tracking_uri)
    if registry_uri:
        mlflow.set_registry_uri(registry_uri)
    return MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri or None)


def promote_alias(
    tracking_uri: str,
    registry_uri: str,
    model_name: str,
    source_alias: str,
    target_alias: str,
) -> None:
    client = _client(tracking_uri=tracking_uri, registry_uri=registry_uri)
    mv = client.get_model_version_by_alias(name=model_name, alias=source_alias)
    client.set_registered_model_alias(name=model_name, alias=target_alias, version=mv.version)
    print(
        f"Alias promoted: {model_name}@{source_alias} (v{mv.version}) -> {model_name}@{target_alias}"
    )


def set_alias(
    tracking_uri: str,
    registry_uri: str,
    model_name: str,
    version: str,
    alias: str,
) -> None:
    client = _client(tracking_uri=tracking_uri, registry_uri=registry_uri)
    client.set_registered_model_alias(name=model_name, alias=alias, version=version)
    print(f"Alias set: {model_name}@{alias} -> v{version}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model Registry alias utilities")
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", ""),
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--registry-uri",
        default=os.getenv("MLFLOW_REGISTRY_URI", ""),
        help="MLflow registry URI",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    promote = sub.add_parser("promote-alias", help="Promote source alias to target alias")
    promote.add_argument("--model-name", default=os.getenv("MODEL_NAME", "artpulse-classifier"))
    promote.add_argument("--source-alias", default="challenger")
    promote.add_argument("--target-alias", default="champion")

    set_alias_cmd = sub.add_parser("set-alias", help="Set alias to explicit model version")
    set_alias_cmd.add_argument("--model-name", default=os.getenv("MODEL_NAME", "artpulse-classifier"))
    set_alias_cmd.add_argument("--version", required=True)
    set_alias_cmd.add_argument("--alias", required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI is required")

    if args.cmd == "promote-alias":
        promote_alias(
            tracking_uri=args.tracking_uri,
            registry_uri=args.registry_uri,
            model_name=args.model_name,
            source_alias=args.source_alias,
            target_alias=args.target_alias,
        )
        return

    if args.cmd == "set-alias":
        set_alias(
            tracking_uri=args.tracking_uri,
            registry_uri=args.registry_uri,
            model_name=args.model_name,
            version=args.version,
            alias=args.alias,
        )
        return

    raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
