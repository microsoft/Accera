####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################
import sys
from azure.cosmos import CosmosClient, PartitionKey
import config
import argparse
from datetime import datetime
import progressbar

HOST = config.settings['host']
MASTER_KEY = config.settings['master_key']
DATABASE_ID = config.settings['database_id']

def get_container(container_name: str, verbose: bool):
    client = CosmosClient(HOST, credential=MASTER_KEY)
    db = client.get_database_client(DATABASE_ID)

    container = db.create_container_if_not_exists(id=container_name, partition_key=PartitionKey(path='/partitionKey'))
    if verbose:
        print('Fetched container with id \'{0}\''.format(container_name))

    return container

def upsert_benchmark_results(result_rows, container_name: str, verbose: bool):
    if verbose:
        print(f"Uploading {len(result_rows)} results to Cosmos DB {DATABASE_ID} in container {container_name}...")
    container = get_container(container_name, verbose)

    for row in result_rows:
        container.upsert_item(row)

    if verbose:
        print("Finished Uploading results to Cosmos DB storage.")

def get_top_result_for_partition(container, partitionKey: str):
    topPerf = list(container.query_items(query="SELECT VALUE MAX(c.TFlops) FROM c", partition_key=partitionKey))
    if len(topPerf) == 0:
        return None

    items = list(container.query_items(
        query="SELECT * From c where c.TFlops=@topPerf",
        partition_key=partitionKey,
        parameters=[
            {"name": "@topPerf", "value": topPerf[0]}
        ]))

    return items[0]

def show_partition_top_result(container, partitionKey: str, print_kernel: bool = False):
    print(f'Top result for partition {partitionKey}:')
    item = get_top_result_for_partition(container, partitionKey)
    if item is None:
        return

    throughput = item['TFlops']
    mma_shape = item['mma_shape']
    use_static_offsets = item['use_static_offsets']
    cache_layout_A = item['cache_layout_A']
    cache_layout_B = item['cache_layout_B']
    cache_strategy_A = item.get('cache_strategy_A', 'NA')
    cache_strategy_B = item.get('cache_strategy_B', 'NA')
    block_tile = item['block_tile']
    k_split = item['k_split']
    double_buffering = item['double_buffering']
    vectorize = item['vectorize']
    num_fused_passes = item['num_fused_passes']
    scheduling_policy = item['scheduling_policy']
    itemId = item['id']
    print(f'{throughput} TFlops')
    print(f'Item id: {itemId}')
    print(f'(Optimizations: MMA shape: {mma_shape}, Static offsets: {use_static_offsets}, CacheA: {cache_layout_A}, CacheB: {cache_layout_B}, StrategyA: {cache_strategy_A}, StrategyB: {cache_strategy_B}, Block tile: {block_tile}, k-split: {k_split}, double buffering: {double_buffering}, vectorize: {vectorize}, fused passes: {num_fused_passes}, scheduling policy: {scheduling_policy})')
    if print_kernel:
        kernel = item['kernelCode']
        print(f'\nKernel code:\n{kernel}')
    print('------------------------------------------')

def show_benchmark_summary(container_name: str):
    container = get_container(container_name, False)
    partitions = list(container.query_items(query="SELECT DISTINCT c.partitionKey from c", enable_cross_partition_query=True))
    partitions = list(list(partitionDict.values())[0] for partitionDict in partitions)
    print(f"Total number of partitions: {len(partitions)}")
    for partitionKey in partitions:
        show_partition_top_result(container, partitionKey)
    else:
        print("--------------Done--------------")

def delete_past_entries(container_name: str, days_to_keep: int):
    container = get_container(container_name, False)
    currentUtcDateTime = list(container.query_items(query="SELECT GetCurrentDateTime() AS currentUtcDateTime"))[0]
    lastDateToKeep = list(container.query_items(query=f"SELECT DateTimeAdd(\"dd\", -{days_to_keep}, \"{currentUtcDateTime['currentUtcDateTime']}\") AS lastDateToKeep"))[0]
    lastDateToKeepStr = lastDateToKeep['lastDateToKeep']
    numTotalItems = list(container.query_items(query=f"SELECT value COUNT(c.id) FROM c", enable_cross_partition_query=True))[0]
    bar = progressbar.ProgressBar(maxval=numTotalItems, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    numItemsDeleted = 0
    for i, item in enumerate(container.query_items(query='SELECT * FROM c', enable_cross_partition_query=True)):
        bar.update(i)
        if datetime.utcfromtimestamp(item['_ts']) < datetime.fromisoformat(lastDateToKeepStr[0:len(lastDateToKeepStr) - 2]):
            numItemsDeleted += 1
            container.delete_item(item, partition_key=item['partitionKey'])
    else:
        bar.finish()
        print(f"Deleted {numItemsDeleted}/{numTotalItems} entries.")

def main(args=[]):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--container', help='The Cosmos DB container to get summary from, e.g. official_build_container_DO_NOT_UPLOAD_HERE', required=True)
    parser.add_argument('-p', '--partition', help='The parition key for which to print the top result, e.g. m3025_n64_k363_a1.0_b0.0_is_os_taFalse_tbFalse_tgAMD_MI100', required=False)
    parser.add_argument('-d', '--days_to_keep', type=int, help='No. of days of most recent data to keep', required=False)

    args = parser.parse_args(args)

    if args.partition:
        show_partition_top_result(get_container(args.container, False), args.partition, True)
    elif args.days_to_keep:
        delete_past_entries(args.container, args.days_to_keep)
    else:
        show_benchmark_summary(args.container)

if __name__ == "__main__":
    main(sys.argv[1:])