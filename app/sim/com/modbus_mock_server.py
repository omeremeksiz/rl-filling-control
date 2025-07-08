from pymodbus.server.async_io import StartAsyncTcpServer
from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext
from pymodbus.datastore import ModbusSequentialDataBlock
from pymodbus.device import ModbusDeviceIdentification
import asyncio

async def run_modbus_mock_server():
    nreg = 200
    # Initialize data store
    store = ModbusSlaveContext(
        di=ModbusSequentialDataBlock(0, [15] * nreg),
        co=ModbusSequentialDataBlock(0, [16] * nreg),
        hr=ModbusSequentialDataBlock(0, [17] * nreg),
        ir=ModbusSequentialDataBlock(0, [18] * nreg)
    )
    context = ModbusServerContext(slaves=store, single=True)

    # Initialize the server identity
    identity = ModbusDeviceIdentification()
    identity.VendorName = 'MockServer'
    identity.ProductCode = 'MBS'
    identity.VendorUrl = 'http://example.com'
    identity.ProductName = 'Modbus Mock Server'
    identity.ModelName = 'Modbus Server'
    identity.MajorMinorRevision = '1.0'

    print("Mock Modbus server running on 127.0.0.1:1502")
    # Start the Modbus server
    await StartAsyncTcpServer(context, identity=identity, address=("127.0.0.1", 1502))

if __name__ == "__main__":
    asyncio.run(run_modbus_mock_server())
