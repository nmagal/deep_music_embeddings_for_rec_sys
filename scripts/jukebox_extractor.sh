#!/usr/bin/env bash
for GENRE in blues classical country disco hiphop jazz metal pop reggae rock
do
	echo $GENRE
	docker run \
		-it \
		--gpus all \
		--rm \
		-v $(pwd)/genres/$GENRE:/input \
		-v $(pwd)/features:/output \
		jukemir/representations_jukebox
done

