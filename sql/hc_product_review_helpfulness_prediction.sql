--https://console.treasuredata.com/app/queries/editor?queryId=1763135

select *
from
(
  select
    a.product_id,
    a.transaction_id,
    b.rating_id as rating_id,
    b.images as image_count,
    a.rating,
    h.true_tag_ids,
    h.true_tag_name,
    h.parent_tag_id,
    h.parent_tag_name,
    h.purchase_count,
    a.comment,
    a.locale,
    a.likely_fake,
    DATEDIFF(CURRENT_TIMESTAMP() ,FROM_UNIXTIME(CAST(a.time as INT))) as days
  from
  (
    select product_id, transaction_id, TD_Last(rating, time) as rating, TD_LAST(comment, time) as comment, TD_LAST(user_locale, time) as locale, td_last(likely_fake, time) as likely_fake, cast(td_last(time, time) as TIMESTAMP) as time
    from sweeper.product_feedback_rating
    group by product_id, transaction_id
  ) a -- product reviews

  -- the following is upvote count
  left join
    (
        select b1.*, e.images
        from
        ( select
          product_id, transaction_id, rating_id, count(upvote) as upvote_count
          from sweeper.product_feedback_rating_upvote
          group by product_id, transaction_id, rating_id
        ) b1
        left join
        (
          select product_id, rating_id, count(image_name) as images
          from sweeper.product_rating_image
          group by product_id, rating_id
        ) e
        on (
          b1.product_id = e.product_id and
          b1.rating_id = e.rating_id
        )
    ) b
    on (
      a.product_id = b.product_id
      and a.transaction_id = b.transaction_id
    )

    -- the following is the true tag id and tag name
  left join
    (
      select
        f.`_id` as product_id,
        f.purchase_count,
        f.true_tag_ids as true_tag_ids,
        g.true_tag_name,
        CASE WHEN g.parent_id = 'None' THEN split(f.true_tag_ids,',')[0] ELSE g.parent_id END as parent_tag_id,
        CASE WHEN g.parent_id = 'None' THEN g.true_tag_name ELSE g.parent_tag_name END as parent_tag_name
      from
        sweeper.contest_20200424120000 f
      left join
        (
          select a.true_tag_id, a.name as true_tag_name, b.true_tag_id as parent_id, b.name as parent_tag_name
          from sweeper.dim_true_tags a
          join sweeper.dim_true_tags b
          on split(a.all_parent_ids,',')[0] = b.true_tag_id

          union ALL

          select true_tag_id, name, 'None' as parent_id, 'None' as parent_name
          from sweeper.dim_true_tags
          where length(all_parent_ids) < 5
        ) g
      on split(f.true_tag_ids,',')[0] = g.true_tag_id
    ) h
    on a.product_id = h.product_id
) full_table
where parent_tag_name = 'Fashion' and locale = 'en' and length(comment) > 0
