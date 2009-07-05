#!/bin/sh

for server in `cat servers.txt`
  do ssh $server "screen -dmS ipe ~/local/bin/ipengine"
done
