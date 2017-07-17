%% --------------MIW-Make up in wild-----------------

load('Makeup_MIW.mat');

for i = 1 : size(MIW_label,1)
    img=reshape(MIW_matrix(:,i),[150 130 3]);
    if MIW_label(i)
        imwrite(img,['/home/dell/Desktop/MU_ds_0pool/MIW/MU/' num2str(i) '.jpg'], 'jpg');
    else
        imwrite(img,['/home/dell/Desktop/MU_ds_0pool/MIW/NM/' num2str(i) '.jpg'], 'jpg');
    end
end

%% -------------VMU----------

load('Makeup_VMU.mat');

for i = 1 : size(VMU_filenames,2)
    img=reshape(VMU_matrix(:,i),[150 130 3]);
    f_name = VMU_filenames{i};
    if strcmp(f_name(end-4),'l')
        imwrite(img,['/home/dell/Desktop/MU_ds_0pool/VMU/MU/LS/' f_name(1:end-6) '.jpg'], 'jpg');
    else if strcmp(f_name((end-4)),'e')
            imwrite(img,['/home/dell/Desktop/MU_ds_0pool/VMU/MU/EM/' f_name(1:end-6) '.jpg'], 'jpg');
        else if strcmp(f_name((end-5):(end-4)),'mu')
                imwrite(img,['/home/dell/Desktop/MU_ds_0pool/VMU/MU/FM/' f_name(1:end-6) '.jpg'], 'jpg');
            else
                imwrite(img,['/home/dell/Desktop/MU_ds_0pool/VMU/NM/' f_name(1:end-4) '.jpg'], 'jpg');
            end
        end
    end
end

%% ------------YMU---------------

load('Makeup_YMU.mat');

for i = 1 : size(YMU_filenames,2)
    img=reshape(YMU_matrix(:,i),[150 130 3]);
    f_name = YMU_filenames{i};
    if strcmp(f_name(end-4),'y')
        if strcmp(f_name((end-6)),'1')
            imwrite(img,['/home/dell/Desktop/MU_ds_0pool/YMU/MU/S1/' f_name(1:end-8) '.jpg'], 'jpg');
        else
            imwrite(img,['/home/dell/Desktop/MU_ds_0pool/YMU/MU/S2/' f_name(1:end-8) '.jpg'], 'jpg');
        end
    else if strcmp(f_name((end-6)),'1')
            imwrite(img,['/home/dell/Desktop/MU_ds_0pool/YMU/NM/S1/' f_name(1:end-8) '.jpg'], 'jpg');
        else
            imwrite(img,['/home/dell/Desktop/MU_ds_0pool/YMU/NM/S2/' f_name(1:end-8) '.jpg'], 'jpg');
        end
    end
    
end