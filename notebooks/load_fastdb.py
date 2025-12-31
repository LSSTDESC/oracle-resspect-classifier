import sys
sys.path.insert( 0, '/global/cfs/cdirs/lsst/groups/TD/SOFTWARE/fastdb_deployment/fastdb_client' )

from fastdb_client import FASTDBClient

fdb = FASTDBClient('dp1')

# testing out the processing versions
res = fdb.post('/getprocvers')
print(res)