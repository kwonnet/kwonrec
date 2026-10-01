-- Run against kwonserver's PostgreSQL database after reviewing on staging.
-- Triggers enqueue IDs only; consumers re-read current authoritative state.
BEGIN;
CREATE SCHEMA IF NOT EXISTS kwonrec;
CREATE TABLE IF NOT EXISTS kwonrec.outbox (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    kind text NOT NULL CHECK (kind IN ('post', 'interaction')),
    entity_id text NOT NULL,
    source_key text UNIQUE,
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    available_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    processed_at timestamptz,
    attempts integer NOT NULL DEFAULT 0,
    dead_at timestamptz,
    error text
);
CREATE INDEX IF NOT EXISTS outbox_pending ON kwonrec.outbox (available_at, id)
    WHERE processed_at IS NULL AND dead_at IS NULL;
CREATE INDEX IF NOT EXISTS outbox_processed ON kwonrec.outbox (processed_at)
    WHERE processed_at IS NOT NULL;

CREATE OR REPLACE FUNCTION kwonrec.capture_post() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE row_data jsonb;
BEGIN
    IF TG_OP = 'DELETE' THEN row_data := to_jsonb(OLD); ELSE row_data := to_jsonb(NEW); END IF;
    -- Counters change frequently: don't reindex for every counter update.
    IF TG_OP = 'UPDATE' AND
       (to_jsonb(OLD) - ARRAY['updatedAt','totalViews','totalLikes','totalReplies','totalShares',
          'totalBookmarks','totalReposts','totalQuotes','totalImpressions','totalTips']) =
       (row_data - ARRAY['updatedAt','totalViews','totalLikes','totalReplies','totalShares',
          'totalBookmarks','totalReposts','totalQuotes','totalImpressions','totalTips']) THEN
       RETURN NEW;
    END IF;
    INSERT INTO kwonrec.outbox(kind, entity_id, payload)
    VALUES ('post', row_data->>'id', jsonb_build_object('author_id',row_data->>'userId',
        'created_at',row_data->>'createdAt'));
    IF TG_OP = 'INSERT' AND row_data->>'parentId' IS NOT NULL AND row_data->>'kind' IN ('REPLY','REPOST','QUOTE') THEN
       INSERT INTO kwonrec.outbox(kind, entity_id, source_key, payload)
       VALUES ('interaction', row_data->>'parentId', 'Post:' || (row_data->>'id'), jsonb_build_object(
           'user_id', row_data->>'userId', 'type', lower(row_data->>'kind'),
           'occurred_at', row_data->>'createdAt', 'duration_seconds', 0)) ON CONFLICT (source_key) DO NOTHING;
    END IF;
    RETURN NULL;
END $$;
DROP TRIGGER IF EXISTS kwonrec_post ON "Post";
CREATE TRIGGER kwonrec_post AFTER INSERT OR UPDATE OR DELETE ON "Post"
FOR EACH ROW EXECUTE FUNCTION kwonrec.capture_post();

CREATE OR REPLACE FUNCTION kwonrec.capture_interaction() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE row_data jsonb := to_jsonb(NEW); uid text;
BEGIN
    uid := COALESCE(row_data->>'userId', row_data->>'senderId');
    IF uid IS NOT NULL AND row_data->>'postId' IS NOT NULL THEN
        INSERT INTO kwonrec.outbox(kind, entity_id, source_key, payload)
        VALUES ('interaction', row_data->>'postId', TG_TABLE_NAME || ':' || (row_data->>'id'), jsonb_build_object(
          'user_id', uid, 'type', TG_ARGV[0],
          'occurred_at', COALESCE(row_data->>'createdAt', row_data->>'timestamp'),
          'duration_seconds', COALESCE((row_data->>'duration')::double precision,0))) ON CONFLICT (source_key) DO NOTHING;
    END IF;
    RETURN NULL;
END $$;

DO $$
DECLARE entry record;
BEGIN
  FOR entry IN SELECT * FROM (VALUES
      ('LikedPost','like'), ('Bookmark','bookmark'), ('PostClick','click'),
      ('PostView','view'), ('PostImpression','impression'), ('PostShare','share'),
      ('PostTip','tip'), ('PostDisinterest','dislike'), ('PostReport','report')
  ) AS sources(table_name,event_type) LOOP
    EXECUTE format('DROP TRIGGER IF EXISTS kwonrec_interaction ON %I',entry.table_name);
    EXECUTE format('CREATE TRIGGER kwonrec_interaction AFTER INSERT ON %I FOR EACH ROW EXECUTE FUNCTION kwonrec.capture_interaction(%L)',
                   entry.table_name,entry.event_type);
  END LOOP;
END $$;
COMMIT;
