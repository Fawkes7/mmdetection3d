category = 'Bag'
level_id = 1
# stat_in_fn = "/home/haoyuan/data/h5PartNet/stats/after_merging2_label_ids/Bag-level-1.txt"
# with open(stat_in_fn, 'r') as fin:
#     part_name_list = [item.rstrip().split()[1] for item in fin.readlines()]
# print('Part Name List: ', part_name_list)
stat_in_fn = f"/home/haoyuan/data/h5PartNet/stats/after_merging2_label_ids/{category}-level-{level_id}.txt"
print('Reading from ', stat_in_fn)
with open(stat_in_fn, 'r') as fin:
    part_name_list = [item.rstrip().split()[1] for item in fin.readlines()]
print('Part Name List: ', part_name_list)