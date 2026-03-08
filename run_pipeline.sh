#!/usr/bin/env bash
# run_pipeline.sh — Compile and submit pipelines for Federated Digital Twin

set -e # Exit on error

PIPELINE_ARG=${1:-"all"}

run_pipeline() {
    local PIPELINE_TYPE=$1
    echo "============================================"
    echo "🎯 Running Strategy: $PIPELINE_TYPE"
    echo "============================================"

    # Determine paths
    if [ "$PIPELINE_TYPE" == "fl" ]; then
        PIPELINE_FILE="src/pipelines/fl_pipeline.py"
        PIPELINE_YAML="pipeline_specs/fl_pipeline.yaml"
    elif [ "$PIPELINE_TYPE" == "single" ]; then
        PIPELINE_FILE="src/pipelines/single_pipeline.py"
        PIPELINE_YAML="pipeline_specs/single_pipeline.yaml"
    else
        PIPELINE_FILE="src/pipelines/${PIPELINE_TYPE}_pipeline.py"
        PIPELINE_YAML="pipeline_specs/${PIPELINE_TYPE}_pipeline.yaml"
    fi

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
kubectl wait --for=condition=ready pod -l app=ml-pipeline -n kubeflow --timeout=300s
kubectl wait --for=condition=ready pod -l app=ml-pipeline-ui -n kubeflow --timeout=300s

echo "   Waiting for API to respond (checking cluster service)..."
for i in {1..60}; do
    # Check the internal cluster service IP or NodePort directly if possible.
    # We will assume that since we removed the localhost port-forwarding, the python
    # kfp Client will be configured to connect to the right host.
    if kubectl get pods -n kubeflow -l app=ml-pipeline-ui | grep -q Running; then
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
if [ "$PIPELINE_ARG" == "all" ]; then
    echo "🔁 Running ALL pipelines..."
    # Ensure generated pipelines are ready
    if [ ! -f "src/pipelines/fl_visual_pipeline.py" ]; then
          echo "⚠️ fl_visual_pipeline.py missing. Generating..."
          uv run python src/pipelines/generate_visual_pipeline.py
    fi

    # Order: Single -> Single Visual -> FL -> FL Visual
    for pt in "single" "single_visual" "fl" "fl_visual"; do
        run_pipeline "$pt"
    done
else
    # Auto-generate if visual and missing (just in case)
    if [ "$PIPELINE_ARG" == "fl_visual" ] && [ ! -f "src/pipelines/fl_visual_pipeline.py" ]; then
        uv run python src/pipelines/generate_visual_pipeline.py
    fi

    run_pipeline "$PIPELINE_ARG"
fi

# 7. Generate Visualizations
echo "📊 Generating Comparison Plots..."
uv run python src/analysis/compare_results.py || echo "⚠️ Could not generate comparison plot (maybe missing data?)"
uv run python src/analysis/worker_diversity.py || echo "⚠️ Could not generate worker diversity plot"

# Generate generalization gap analysis for both pipeline types
if [ "$PIPELINE_ARG" == "single" ] || [ "$PIPELINE_ARG" == "single_visual" ] || [ "$PIPELINE_ARG" == "all" ]; then
    uv run python src/analysis/generalization_gap.py single || echo "⚠️ Could not generate single generalization gap"
fi
if [ "$PIPELINE_ARG" == "fl" ] || [ "$PIPELINE_ARG" == "fl_visual" ] || [ "$PIPELINE_ARG" == "all" ]; then
    uv run python src/analysis/generalization_gap.py fl || echo "⚠️ Could not generate FL generalization gap"
fi

echo ""
echo "🎉 Process Complete!"
