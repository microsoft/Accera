####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################
import os

settings = {
    'host': os.environ.get('ACCOUNT_HOST', '...'),
    'master_key': os.environ.get('ACCOUNT_KEY', '...'),
    'database_id': os.environ.get('COSMOS_DATABASE', 'perf-results'),
}