import argparse
from omegaconf import OmegaConf
from appfl.agent import APPFLServerAgent
from appfl.comm.grpc import GRPCServerCommunicator, serve

import socket
import sys
def get_ip_address():
    try:
        # create a socket object
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # try to connect to an impossible IP and port
        s.connect(("10.255.255.255", 1))
        IP = s.getsockname()[0]
    except Exception as e:
        IP = "127.0.0.1"  # fallback to localhost
    finally:
        s.close()
    return IP

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--config", 
    type=str, 
    default="config/server_fedavg.yaml",
    help="Path to the configuration file."
)
args = argparser.parse_args()

server_agent_config = OmegaConf.load(args.config)

IP = get_ip_address()
port = 6379
server_agent_config.server_configs.comm_configs.grpc_configs.server_uri = IP + ":" + str(port)
print("\n URI of server: {}:{}".format(IP,port),file=sys.stderr)
print("\n URI of server: {}:{}".format(IP,port),file=sys.stdout)
print("\n URI of server: {}:{}".format(IP,port))


server_agent = APPFLServerAgent(server_agent_config=server_agent_config)

communicator = GRPCServerCommunicator(
    server_agent,
    max_message_size=server_agent_config.server_configs.comm_configs.grpc_configs.max_message_size,
    logger=server_agent.logger,
)

serve(
    communicator,
    **server_agent_config.server_configs.comm_configs.grpc_configs,
)
