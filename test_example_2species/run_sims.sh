#!/bin/bash

$(MPIRUN_COMMAND) $(STELLA_MASTER_BIN) example.in

# Copy to results dir
cp input.* results_master 

# Run the same sim on the feature branch we want to test
$(MPIRUN_COMMAND) $(STELLA_FEATURE_BIN) example.in

# Copy to the feature dir.
cp input.* $(RESULTS_FEATURE_DIR) 

