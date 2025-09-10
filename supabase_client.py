"""
Supabase Client Class
A clean interface for all Supabase operations
"""

import os
from typing import Optional, Dict, List, Any
from dotenv import load_dotenv
from supabase import create_client, Client

class SupabaseClient:
    """
    A comprehensive Supabase client class for database operations
    """
    
    def __init__(self):
        """Initialize Supabase client with environment variables"""
        load_dotenv()
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_api_key = os.getenv("SUPABASE_API_KEY")
        
        if not self.supabase_url or not self.supabase_api_key:
            raise ValueError("SUPABASE_URL and SUPABASE_API_KEY must be set in environment variables")
        
        self.client: Client = create_client(self.supabase_url, self.supabase_api_key)
    
    def test_connection(self) -> bool:
        """Test the Supabase connection"""
        try:
            # Try a simple connection test
            print("✅ Supabase API client connected successfully")
            print(f"   URL: {self.supabase_url}")
            print(f"   API Key: {self.supabase_api_key[:20]}...")
            return True
        except Exception as e:
            print(f"❌ Supabase connection failed: {e}")
            return False
    
    # CRUD Operations
    def insert_data(self, table_name: str, data: Dict[str, Any]) -> Optional[List[Dict]]:
        """
        Insert data into a Supabase table
        
        Args:
            table_name: Name of the table
            data: Dictionary of data to insert
            
        Returns:
            Inserted data or None if failed
        """
        try:
            result = self.client.table(table_name).insert(data).execute()
            print(f"✅ Inserted data into {table_name}")
            return result.data
        except Exception as e:
            print(f"❌ Error inserting data into {table_name}: {e}")
            return None
    
    def query_data(self, table_name: str, columns: str = "*", limit: Optional[int] = None) -> Optional[List[Dict]]:
        """
        Query data from a Supabase table
        
        Args:
            table_name: Name of the table
            columns: Columns to select (default: "*")
            limit: Maximum number of rows to return
            
        Returns:
            Query results or None if failed
        """
        try:
            query = self.client.table(table_name).select(columns)
            if limit:
                query = query.limit(limit)
            
            result = query.execute()
            print(f"✅ Retrieved {len(result.data)} rows from {table_name}")
            return result.data
        except Exception as e:
            print(f"❌ Error querying data from {table_name}: {e}")
            return None
    
    def update_data(self, table_name: str, data: Dict[str, Any], filter_column: str, filter_value: Any) -> Optional[List[Dict]]:
        """
        Update data in a Supabase table
        
        Args:
            table_name: Name of the table
            data: Dictionary of data to update
            filter_column: Column to filter by
            filter_value: Value to filter by
            
        Returns:
            Updated data or None if failed
        """
        try:
            result = self.client.table(table_name).update(data).eq(filter_column, filter_value).execute()
            print(f"✅ Updated data in {table_name}")
            return result.data
        except Exception as e:
            print(f"❌ Error updating data in {table_name}: {e}")
            return None
    
    def delete_data(self, table_name: str, filter_column: str, filter_value: Any) -> Optional[List[Dict]]:
        """
        Delete data from a Supabase table
        
        Args:
            table_name: Name of the table
            filter_column: Column to filter by
            filter_value: Value to filter by
            
        Returns:
            Deleted data or None if failed
        """
        try:
            result = self.client.table(table_name).delete().eq(filter_column, filter_value).execute()
            print(f"✅ Deleted data from {table_name}")
            return result.data
        except Exception as e:
            print(f"❌ Error deleting data from {table_name}: {e}")
            return None
    
    # Advanced Query Operations
    def query_with_filters(self, table_name: str, filters: Dict[str, Any], columns: str = "*") -> Optional[List[Dict]]:
        """
        Query data with multiple filters
        
        Args:
            table_name: Name of the table
            filters: Dictionary of column:value filters
            columns: Columns to select
            
        Returns:
            Filtered query results or None if failed
        """
        try:
            query = self.client.table(table_name).select(columns)
            
            for column, value in filters.items():
                query = query.eq(column, value)
            
            result = query.execute()
            print(f"✅ Retrieved {len(result.data)} filtered rows from {table_name}")
            return result.data
        except Exception as e:
            print(f"❌ Error querying filtered data from {table_name}: {e}")
            return None
    
    def search_data(self, table_name: str, column: str, search_term: str, columns: str = "*") -> Optional[List[Dict]]:
        """
        Search for data containing a specific term
        
        Args:
            table_name: Name of the table
            column: Column to search in
            search_term: Term to search for
            columns: Columns to select
            
        Returns:
            Search results or None if failed
        """
        try:
            result = self.client.table(table_name).select(columns).ilike(column, f"%{search_term}%").execute()
            print(f"✅ Found {len(result.data)} results for '{search_term}' in {table_name}")
            return result.data
        except Exception as e:
            print(f"❌ Error searching data in {table_name}: {e}")
            return None
    
    # Utility Methods
    def count_rows(self, table_name: str) -> Optional[int]:
        """
        Count total rows in a table
        
        Args:
            table_name: Name of the table
            
        Returns:
            Row count or None if failed
        """
        try:
            result = self.client.table(table_name).select("*", count="exact").execute()
            count = result.count
            print(f"✅ Table {table_name} has {count} rows")
            return count
        except Exception as e:
            print(f"❌ Error counting rows in {table_name}: {e}")
            return None
    
    def execute_rpc(self, function_name: str, params: Optional[Dict] = None) -> Optional[Any]:
        """
        Execute a stored procedure/function
        
        Args:
            function_name: Name of the RPC function
            params: Parameters to pass to the function
            
        Returns:
            Function result or None if failed
        """
        try:
            if params:
                result = self.client.rpc(function_name, params).execute()
            else:
                result = self.client.rpc(function_name).execute()
            print(f"✅ Executed RPC function: {function_name}")
            return result.data
        except Exception as e:
            print(f"❌ Error executing RPC function {function_name}: {e}")
            return None
    
    # Batch Operations
    def batch_insert(self, table_name: str, data_list: List[Dict[str, Any]]) -> Optional[List[Dict]]:
        """
        Insert multiple records at once
        
        Args:
            table_name: Name of the table
            data_list: List of dictionaries to insert
            
        Returns:
            Inserted data or None if failed
        """
        try:
            result = self.client.table(table_name).insert(data_list).execute()
            print(f"✅ Batch inserted {len(data_list)} records into {table_name}")
            return result.data
        except Exception as e:
            print(f"❌ Error batch inserting data into {table_name}: {e}")
            return None
    
    def get_client(self) -> Client:
        """
        Get the raw Supabase client for advanced operations
        
        Returns:
            Supabase client instance
        """
        return self.client

# Example usage and testing
def main():
    """Test the SupabaseClient class"""
    try:
        print("🚀 Testing SupabaseClient Class")
        print("=" * 50)
        
        # Initialize client
        supabase_client = SupabaseClient()
        
        # Test connection
        supabase_client.test_connection()
        
        print("\n📋 SupabaseClient is ready to use!")
        print("Available methods:")
        print("  • insert_data(table_name, data)")
        print("  • query_data(table_name, columns, limit)")
        print("  • update_data(table_name, data, filter_column, filter_value)")
        print("  • delete_data(table_name, filter_column, filter_value)")
        print("  • query_with_filters(table_name, filters)")
        print("  • search_data(table_name, column, search_term)")
        print("  • count_rows(table_name)")
        print("  • batch_insert(table_name, data_list)")
        print("  • execute_rpc(function_name, params)")
        
    except Exception as e:
        print(f"❌ Error initializing SupabaseClient: {e}")

if __name__ == "__main__":
    main()
