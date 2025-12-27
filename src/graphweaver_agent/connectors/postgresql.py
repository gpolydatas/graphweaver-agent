"""PostgreSQL Connector"""
import re
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor

from graphweaver_agent.models import (
    DataSourceConfig, DatabaseType, ColumnDataType,
    ColumnMetadata, ColumnStatistics, ExistingForeignKey, TableMetadata,
)


class PostgreSQLConnector:
    TYPE_MAP = {
        'integer': ColumnDataType.INTEGER, 'int': ColumnDataType.INTEGER,
        'int4': ColumnDataType.INTEGER, 'bigint': ColumnDataType.BIGINT,
        'int8': ColumnDataType.BIGINT, 'smallint': ColumnDataType.SMALLINT,
        'serial': ColumnDataType.INTEGER, 'bigserial': ColumnDataType.BIGINT,
        'decimal': ColumnDataType.DECIMAL, 'numeric': ColumnDataType.DECIMAL,
        'real': ColumnDataType.FLOAT, 'double precision': ColumnDataType.FLOAT,
        'varchar': ColumnDataType.VARCHAR, 'character varying': ColumnDataType.VARCHAR,
        'text': ColumnDataType.TEXT, 'boolean': ColumnDataType.BOOLEAN,
        'bool': ColumnDataType.BOOLEAN, 'date': ColumnDataType.DATE,
        'timestamp': ColumnDataType.TIMESTAMP, 'uuid': ColumnDataType.UUID,
        'json': ColumnDataType.JSON, 'jsonb': ColumnDataType.JSON,
    }
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.db_type = config.db_type
    
    @contextmanager
    def connection(self):
        conn = psycopg2.connect(
            host=self.config.host, port=self.config.port,
            database=self.config.database, user=self.config.username,
            password=self.config.password,
        )
        try:
            yield conn
        finally:
            conn.close()
    
    def test_connection(self) -> Dict[str, Any]:
        try:
            with self.connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT version(), current_database(), current_user")
                    row = cur.fetchone()
                    return {"success": True, "database": row["current_database"], "user": row["current_user"]}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def map_data_type(self, db_type: str) -> ColumnDataType:
        base = re.sub(r'\(.*\)', '', db_type.lower().strip())
        return self.TYPE_MAP.get(base, ColumnDataType.UNKNOWN)
    
    def get_tables(self) -> List[str]:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = %s AND table_type = 'BASE TABLE'
                """, (self.config.schema_name,))
                return [row[0] for row in cur.fetchall()]
    
    def get_tables_with_info(self) -> List[Dict[str, Any]]:
        with self.connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT t.table_name,
                        (SELECT count(*) FROM information_schema.columns c 
                         WHERE c.table_schema = t.table_schema AND c.table_name = t.table_name) as column_count,
                        COALESCE(s.n_live_tup, 0) as row_estimate
                    FROM information_schema.tables t
                    LEFT JOIN pg_stat_user_tables s ON s.schemaname = t.table_schema AND s.relname = t.table_name
                    WHERE t.table_schema = %s AND t.table_type = 'BASE TABLE'
                """, (self.config.schema_name,))
                return [dict(row) for row in cur.fetchall()]
    
    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get schema information for a table."""
        with self.connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s
                    ORDER BY ordinal_position
                """, (self.config.schema_name, table_name))
                columns = [dict(row) for row in cur.fetchall()]
                
                cur.execute("""
                    SELECT a.attname FROM pg_index i
                    JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
                    JOIN pg_class c ON c.oid = i.indrelid
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = %s AND c.relname = %s AND i.indisprimary
                """, (self.config.schema_name, table_name))
                primary_keys = [r["attname"] for r in cur.fetchall()]
                
                cur.execute(f'SELECT COUNT(*) as cnt FROM "{self.config.schema_name}"."{table_name}"')
                row_count = cur.fetchone()["cnt"]
                
                return {
                    "table_name": table_name,
                    "columns": columns,
                    "primary_keys": primary_keys,
                    "row_count": row_count,
                }
    
    def get_table_metadata(self, table_name: str) -> TableMetadata:
        with self.connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT column_name, data_type, is_nullable FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position
                """, (self.config.schema_name, table_name))
                columns = [ColumnMetadata(
                    table_name=table_name, column_name=r["column_name"],
                    data_type=self.map_data_type(r["data_type"]),
                    is_nullable=r["is_nullable"] == "YES"
                ) for r in cur.fetchall()]
                
                cur.execute("""
                    SELECT a.attname FROM pg_index i
                    JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
                    JOIN pg_class c ON c.oid = i.indrelid
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = %s AND c.relname = %s AND i.indisprimary
                """, (self.config.schema_name, table_name))
                pk_cols = [r["attname"] for r in cur.fetchall()]
                
                cur.execute(f'SELECT COUNT(*) as cnt FROM "{self.config.schema_name}"."{table_name}"')
                row = cur.fetchone()
                row_count = row["cnt"] if row else 0
                
                for col in columns:
                    if col.column_name in pk_cols:
                        col.is_primary_key = True
                        col.is_unique = True
                
                return TableMetadata(
                    schema_name=self.config.schema_name, table_name=table_name,
                    row_count=row_count, columns=columns, primary_key_columns=pk_cols,
                )
    
    def get_column_statistics(self, table_name: str, column_name: str) -> ColumnStatistics:
        with self.connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f'''
                    SELECT COUNT(*) as total, COUNT(DISTINCT "{column_name}") as distinct_count,
                           COUNT(*) - COUNT("{column_name}") as null_count
                    FROM "{self.config.schema_name}"."{table_name}"
                ''')
                counts = cur.fetchone()
                total = counts["total"] or 0
                distinct = counts["distinct_count"] or 0
                nulls = counts["null_count"] or 0
                
                cur.execute(f'''
                    SELECT DISTINCT "{column_name}" as val
                    FROM "{self.config.schema_name}"."{table_name}"
                    WHERE "{column_name}" IS NOT NULL LIMIT 10
                ''')
                samples = [r["val"] for r in cur.fetchall()]
                
                return ColumnStatistics(
                    table_name=table_name, column_name=column_name,
                    total_count=total, distinct_count=distinct, null_count=nulls,
                    uniqueness_ratio=distinct/total if total > 0 else 0,
                    null_ratio=nulls/total if total > 0 else 0,
                    sample_values=samples,
                )
    
    def validate_fk(self, source_table: str, source_column: str,
                    target_table: str, target_column: str) -> Dict[str, Any]:
        """Validate if a FK relationship is valid by checking data."""
        with self.connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f'''
                    SELECT COUNT(*) as total
                    FROM "{self.config.schema_name}"."{source_table}"
                    WHERE "{source_column}" IS NOT NULL
                ''')
                total = cur.fetchone()["total"]
                
                if total == 0:
                    return {
                        "is_valid": False, "match_rate": 0, "matched_count": 0,
                        "total_count": 0, "orphan_count": 0, "cardinality": "unknown",
                        "reason": "No data in source column"
                    }
                
                cur.execute(f'''
                    SELECT COUNT(*) as matched
                    FROM "{self.config.schema_name}"."{source_table}" s
                    WHERE s."{source_column}" IS NOT NULL
                      AND EXISTS (
                          SELECT 1 FROM "{self.config.schema_name}"."{target_table}" t
                          WHERE t."{target_column}" = s."{source_column}"
                      )
                ''')
                matched = cur.fetchone()["matched"]
                match_rate = matched / total if total > 0 else 0
                
                cur.execute(f'''
                    SELECT COUNT(*) as src_count, COUNT(DISTINCT "{source_column}") as distinct_count
                    FROM "{self.config.schema_name}"."{source_table}"
                    WHERE "{source_column}" IS NOT NULL
                ''')
                card_row = cur.fetchone()
                cardinality = "1:1" if card_row["src_count"] == card_row["distinct_count"] else "1:N"
                
                return {
                    "is_valid": match_rate >= 0.95,
                    "match_rate": match_rate,
                    "matched_count": matched,
                    "total_count": total,
                    "orphan_count": total - matched,
                    "cardinality": cardinality,
                }
    
    def check_referential_integrity(self, source_table: str, source_column: str,
                                     target_table: str, target_column: str,
                                     sample_size: int = 10000) -> Dict[str, Any]:
        with self.connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f'''
                    SELECT DISTINCT "{source_column}" as val
                    FROM "{self.config.schema_name}"."{source_table}"
                    WHERE "{source_column}" IS NOT NULL LIMIT %s
                ''', (sample_size,))
                source_values = [r["val"] for r in cur.fetchall()]
                
                if not source_values:
                    return {"match_rate": 0.0, "sample_size": 0, "matches": 0}
                
                cur.execute(f'''
                    SELECT COUNT(DISTINCT "{target_column}") as matches
                    FROM "{self.config.schema_name}"."{target_table}"
                    WHERE "{target_column}" = ANY(%s)
                ''', (source_values,))
                matches = cur.fetchone()["matches"]
                
                return {
                    "match_rate": matches / len(source_values),
                    "sample_size": len(source_values),
                    "matches": matches,
                    "orphans": len(source_values) - matches,
                }
    
    def execute_query(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        if not query.strip().lower().startswith("select"):
            raise ValueError("Only SELECT queries allowed")
        if "limit" not in query.lower():
            query = f"{query} LIMIT {limit}"
        with self.connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query)
                return [dict(r) for r in cur.fetchall()]


def create_connector(config: DataSourceConfig):
    if config.db_type == DatabaseType.POSTGRESQL:
        return PostgreSQLConnector(config)
    raise ValueError(f"Unsupported: {config.db_type}")
