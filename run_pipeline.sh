#!/usr/bin/env bash
# run_pipeline.sh — Compile and submit pipelines for Federated Digital Twin
# Ensure a rich PATH regardless of how the script is invoked (sh vs bash)
export PATH="/Users/supat/.local/bin:/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/local/sbin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

set -e # Exit on error

PIPELINE_ARG_RAW=${1:-"all"}
# Portable hyphen to underscore replacement
PIPELINE_ARG=$(echo "$PIPELINE_ARG_RAW" | tr '-' '_')

run_pipeline() {
    local PIPELINE_TYPE=$1
    echo "============================================"
    echo "🎯 Running Strategy: $PIPELINE_TYPE"
    echo "============================================"

    # Determine paths
    PIPELINE_FILE="src/pipelines/${PIPELINE_TYPE}_pipeline.py"
    PIPELINE_YAML="pipeline_specs/${PIPELINE_TYPE}_pipeline.yaml"

    echo "📄 Install dependencies & Compiling Pipeline for $PIPELINE_TYPE..."
    uv pip install kfp boto3 --quiet || echo "Warning: pip install failed"

    if [ -f "$PIPELINE_FILE" ]; then
        uv run python "$PIPELINE_FILE"
        echo "✅ Generated $PIPELINE_YAML"
    else
        echo "❌ Pipeline file '$PIPELINE_FILE' not found!"
        return 1
    fi
    
    echo "   Submitting Pipeline Run..."
    uv run python src/automate_run.py "$PIPELINE_TYPE"
    
    if [ $? -eq 0 ]; then
        echo "✅ Pipeline Run Completed."
        echo "📥 Fetching Results..."
        uv run python src/fetch_results.py "$PIPELINE_TYPE"
        
        # Display Results
        if [ -f "metrics/last_run_id_$PIPELINE_TYPE.txt" ]; then
            RUN_ID=$(cat "metrics/last_run_id_$PIPELINE_TYPE.txt")
            echo "Last Run ID for $PIPELINE_TYPE: $RUN_ID"
            METRICS_FILE="metrics/metrics_${PIPELINE_TYPE}_${RUN_ID}.csv"
            if [ -f "$METRICS_FILE" ]; then
                echo "✅ Results saved to: $METRICS_FILE"
                echo "📊 Preview (last 5 lines):"
                tail -n 5 "$METRICS_FILE"
            fi
        fi
    else
        echo "❌ Pipeline Run Failed."
        return 1
    fi
}

echo "🧪 Preparing Execution Environment..."
KUBECTL="/opt/homebrew/bin/kubectl"
if [ ! -f "$KUBECTL" ]; then KUBECTL="/usr/local/bin/kubectl"; fi

$KUBECTL wait --for=condition=ready pod -l app=ml-pipeline -n kubeflow --timeout=300s
$KUBECTL wait --for=condition=ready pod -l app=ml-pipeline-ui -n kubeflow --timeout=300s

echo "   Waiting for API to respond (checking cluster service)..."
for i in {1..60}; do
    if $KUBECTL get pods -n kubeflow -l app=ml-pipeline-ui | grep -q Running; then
        echo "✅ KFP API is UP."
        break
    fi
    if [ $i -eq 60 ]; then
        echo "❌ KFP API failed to start."
        exit 1
    fi
    sleep 2
done

# Run Logic
if [ "$PIPELINE_ARG" == "all_single_cluster" ]; then
    echo "🔁 Running ALL Single Cluster pipelines..."
    # Ensure generated pipelines are ready and up-to-date with config
    echo "⚠️ Generating/Updating fed_twin_visual_single_cluster_pipeline.py..."
    uv run python src/pipelines/generate_fed_twin_visual_pipeline.py

    # Order: Single Twin -> Single Twin Visual -> Fed Twin -> Fed Twin Visual
    for pt in "single_twin_single_cluster" "single_twin_visual_single_cluster" "fed_twin_single_cluster" "fed_twin_visual_single_cluster"; do
        run_pipeline "$pt"
    done
elif [ "$PIPELINE_ARG" == "all_multi_cluster" ]; then
    echo "🔁 Running ALL Multi Cluster pipelines..."
    # Order: Single Twin Multi Cluster -> Fed Twin Multi Cluster
    for pt in "single_twin_multi_cluster" "fed_twin_multi_cluster"; do
        run_pipeline "$pt"
    done
elif [ "$PIPELINE_ARG" == "all" ]; then
    echo "❌ 'all' mode is no longer supported because single_cluster and multi_cluster pipelines require different environments."
    echo "👉 Please use 'all_single_cluster' or 'all_multi_cluster' instead."
    exit 1
else
    # Auto-generate if visual
    if [ "$PIPELINE_ARG" == "fed_twin_visual_single_cluster" ]; then
        echo "⚠️ Generating/Updating fed_twin_visual_single_cluster_pipeline.py..."
        uv run python src/pipelines/generate_fed_twin_visual_pipeline.py
    fi

    run_pipeline "$PIPELINE_ARG"
fi

# 7. Generate Visualizations
echo "📊 Generating Comparison Plots..."
uv run python src/analysis/compare_results.py || echo "⚠️ Could not generate comparison plot (maybe missing data?)"
uv run python src/analysis/worker_diversity.py || echo "⚠️ Could not generate worker diversity plot"

# Generate generalization gap analysis for both pipeline types
if [[ "$PIPELINE_ARG" == *"single_twin"* ]] || [ "$PIPELINE_ARG" == "all_single_cluster" ] || [ "$PIPELINE_ARG" == "all_multi_cluster" ]; then
    if [[ "$PIPELINE_ARG" == *"multi_cluster"* ]] || [ "$PIPELINE_ARG" == "all_multi_cluster" ]; then
        uv run python src/analysis/generalization_gap.py single_twin_multi_cluster || echo "⚠️ Could not generate single_twin_multi_cluster generalization gap"
    else
        uv run python src/analysis/generalization_gap.py single_twin_single_cluster || echo "⚠️ Could not generate single_twin_single_cluster generalization gap"
    fi
fi
if [[ "$PIPELINE_ARG" == *"fed_twin"* ]] || [ "$PIPELINE_ARG" == "all_single_cluster" ] || [ "$PIPELINE_ARG" == "all_multi_cluster" ]; then
    if [[ "$PIPELINE_ARG" == *"multi_cluster"* ]] || [ "$PIPELINE_ARG" == "all_multi_cluster" ]; then
        uv run python src/analysis/generalization_gap.py fed_twin_multi_cluster || echo "⚠️ Could not generate fed_twin_multi_cluster generalization gap"
    else
        uv run python src/analysis/generalization_gap.py fed_twin_single_cluster || echo "⚠️ Could not generate fed_twin_single_cluster generalization gap"
    fi
fi

echo ""
echo "🎉 Process Complete!"
