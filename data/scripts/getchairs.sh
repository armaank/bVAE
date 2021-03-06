#! /bin/sh

cd .. # go upstairs into main data folder

if [ -f "rendered_chairs.tar" ]; then
    tar -xvf rendered_chairs.tar
    root=rendered_chairs
    new_root="3DChairs/images"
    rm $root"/all_chair_names.mat"
    mkdir -p $new_root
    n=1
    for dir in `ls -1t $root`; do
        for imgpath in `ls -1t $root/$dir/renders/*`; do
            imgname=$(echo "$imgpath" | cut -d"/" -f4)
            newpath=$img" "$new_root"/"$n"_"$imgname
            mv $imgpath $newpath
            n=$((n+1))
        done
    done
    rm -rf $root

else
    echo "download 3DChairs dataset."
    wget https://www.di.ens.fr/willow/research/seeing3Dchairs/data/rendered_chairs.tar
    
    tar -xvf rendered_chairs.tar
    root=rendered_chairs
    new_root="3DChairs/images"
    rm $root"/all_chair_names.mat"
    mkdir -p $new_root
    n=1
    for dir in `ls -1t $root`; do
        for imgpath in `ls -1t $root/$dir/renders/*`; do
            imgname=$(echo "$imgpath" | cut -d"/" -f4)
            newpath=$img" "$new_root"/"$n"_"$imgname
            mv $imgpath $newpath
            n=$((n+1))
        done
    done
    rm -rf $root
fi


