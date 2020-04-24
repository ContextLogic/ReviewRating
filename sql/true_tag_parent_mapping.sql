select a.true_tag_id, a.name, b.true_tag_id as parent_id, b.name as parent_name
from sweeper.dim_true_tags a
join sweeper.dim_true_tags b
on split(a.all_parent_ids,',')[0] = b.true_tag_id

union ALL

select true_tag_id, name, 'None' as parent_id, 'None' as parent_name
from sweeper.dim_true_tags
where length(all_parent_ids) < 5

