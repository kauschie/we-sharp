import subprocess
import random
import sys
import time
import logging

# Set up logging
logging.basicConfig(
    filename='pia_server.log', 
    filemode='a',  
    format='%(asctime)s - %(levelname)s - %(message)s', 
    level=logging.INFO  
)

def load_server_list(file_path="server-list.txt"):
    """Load servers from a text file into a list."""
    servers = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                server_data = line.strip().split()
                if len(server_data) == 3:
                    servers.append({
                        'name': server_data[0],
                        'config': server_data[1],
                        'visited': server_data[2].lower() == 'true'
                    })
        logging.info("Successfully loaded server list from %s", file_path)
    except FileNotFoundError:
        logging.error("File %s not found.", file_path)
        sys.exit(1)
    return servers

def save_server_list(servers, file_path="server-list.txt"):
    """Save the updated server list back to the text file."""
    try:
        with open(file_path, 'w') as f:
            for server in servers:
                f.write(f"{server['name']} {server['config']} {server['visited']}\n")
        logging.info("Successfully saved server list to %s", file_path)
    except Exception as e:
        logging.error("Error saving server list: %s", str(e))
        sys.exit(1)

def reset_server_list(servers, file_path="server-list.txt"):
    """Reset all servers to unvisited (False) if all have been visited."""
    for server in servers:
        server['visited'] = False
    save_server_list(servers, file_path)
    logging.info("All servers have been reset to unvisited.")

def get_unvisited_servers(servers):
    """Filter and return the list of unvisited servers."""
    return [server for server in servers if not server['visited']]

def get_current_ip():
    """Returns the current external IP address."""
    try:
        result = subprocess.run(['curl', '-s', 'ifconfig.me'], stdout=subprocess.PIPE, text=True)
        return result.stdout.strip()
    except Exception as e:
        logging.error(f"Error fetching current IP: {e}")
        return None

def wait_for_vpn_connection(old_ip, retries=10, delay=5):
    """Wait for VPN connection by checking if the external IP has changed."""
    for attempt in range(retries):
        current_ip = get_current_ip()
        if current_ip and current_ip != old_ip:
            logging.info(f"VPN connected successfully! New IP: {current_ip}")
            return True
        logging.info(f"Attempt {attempt + 1}/{retries}: VPN not connected yet. Current IP: {current_ip}")
        time.sleep(delay)
    logging.error("Failed to establish VPN connection after retries.")
    return False

def connect_to_server(server):
    """Connect to the selected server using OpenVPN via the run_openvpn.sh helper script."""
    try:
        old_ip = get_current_ip()
        if not old_ip:
            logging.error("Unable to retrieve current IP. Aborting.")
            return False

        # Disconnect from any active VPN connection, but don't fail if there's no active connection
        subprocess.run(['sudo', 'pkill', 'openvpn'], check=False)

        logging.info(f"Connecting to {server['name']}...")

        # Call the helper script to run OpenVPN, detached
        subprocess.Popen(['./run_openvpn.sh', server['config']])

        # Wait for the VPN connection to establish
        if wait_for_vpn_connection(old_ip):
            logging.info(f"Connected to {server['name']} successfully.")
            return True
        else:
            logging.error(f"Failed to connect to {server['name']}")
            return False
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to connect to server {server['name']}: {e}")
        return False

def change_pia_server():
    """Change PIA server by selecting an unvisited server."""
    servers = load_server_list()

    # Get unvisited servers
    unvisited_servers = get_unvisited_servers(servers)

    if not unvisited_servers:
        logging.info("All servers have been visited. Resetting the list...")
        reset_server_list(servers)
        unvisited_servers = get_unvisited_servers(servers)

    # Select a random unvisited server
    selected_server = random.choice(unvisited_servers)
    if connect_to_server(selected_server):
        # Mark the server as visited and save the list
        for server in servers:
            if server['name'] == selected_server['name']:
                server['visited'] = True
                break
        save_server_list(servers)
        logging.info(f"Successfully connected to {selected_server['name']} and marked it as visited.")

if __name__ == "__main__":
    change_pia_server()
