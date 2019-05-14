function [] = process_seqs(varargin)
    % turn off warnings so that the output of the alg. is less cluttered
    warning('off','all'); 

    %root directory that contains dir with segmentation images per sequence
    segmentation_dirs = '<DIR/WHERE/SEGMENTATION/IS/STORED>';  

    %root directory that contains dir with original RGB images per sequence
    sequences_dirs = '<DIR/WHERE/RGB/IS/STORED>';              

    visualize = 0;
    rewrite_vis = 0;
    
    %list of sequences names (must match directory names in segmentation_dirs and sequences_dirs)
    list_file = 'list.txt';   
    if (~isempty(varargin))
        list_file = varargin{1};
    end
    
    fid = fopen(list_file, 'r');
    sequence_list = textscan(fid,'%s','delimiter','\n');
    sequence_list = sequence_list{1};
    fclose(fid);

    num_seq = length(sequence_list);

    % where to store the results
    out_dir = '<OUTPUT/DIRECTORY>';
    
    mkdir(out_dir);
    for i = 1:num_seq
        fprintf('Processing sequence : %s\n', sequence_list{i});
        seq_dir = [out_dir '/' sequence_list{i}];
        
        if (exist([out_dir '/' sequence_list{i} '_optimization.mat'],'file') > 0)
            load([out_dir '/' sequence_list{i} '_optimization.mat']);    
        else
            [bboxes, costs, area_scores, constraints_flags, costs_gt, area_scores_gt] ...
                     = optimize_bboxes([segmentation_dirs '/' sequence_list{i} '/'], ... 
                               [sequences_dirs '/' sequence_list{i} '/groundtruth.txt']);

            save([out_dir '/' sequence_list{i} '_optimization.mat'], 'bboxes', 'costs', 'area_scores', 'constraints_flags', 'costs_gt', 'area_scores_gt');
        end

        if (exist(seq_dir,'dir') > 0 && rewrite_vis == 0)
            continue;
        end
        
        if (visualize > 0)
            mkdir(seq_dir);
            ext = {'*.jpeg', '*.jpg','*.png'};
            images = [];
            for k = 1:length(ext)
                images = [images dir([sequences_dirs '/' sequence_list{i} '/' ext{k}])];
            end
            % make the image name with absolute path
            for k = 1:length(images)
                images(k).name = [sequences_dirs '/' sequence_list{i} '/' images(k).name];
            end                           

            images_seg = [];
            for k = 1:length(ext)
                images_seg = [images_seg dir([segmentation_dirs '/' sequence_list{i} '/' ext{k}])];
            end
            % make the image name with absolute path
            for k = 1:length(images_seg)
                images_seg(k).name = [segmentation_dirs '/' sequence_list{i} '/' images_seg(k).name];
            end                           

            gt = dlmread([sequences_dirs '/' sequence_list{i} '/groundtruth.txt']);
            num_frames = size(bboxes,1);

            img = imread(images(1).name);
            width = 200;
            height = 200;
            for j = 1:num_frames
                [x1, x2, y1, y2] = get_roi(bboxes(j,:), [size(img, 2) size(img,1)]); 
                w = x2-x1;
                h = y2-y1;
                if (w > width)
                    width = w;
                end
                if (h > height)
                   height = h;  
                end
            end
            

           for j = 1:num_frames
               img = imread(images(j).name);
               seg = imread(images_seg(j).name);

               [x1, x2, y1, y2] = get_roi(bboxes(j,:), [size(img, 2) size(img,1)]); 
               x_min = max([round((x1+x2)/2 - width/2) 1]);
               x_max = min([round((x1+x2)/2 + width/2) size(img,2)]);
               y_min = max([round((y1+y2)/2 - height/2) 1]);
               y_max = min([round((y1+y2)/2 + height/2) size(img,1)]);

               roi_x = x_min:x_max;
               roi_y = y_min:y_max; 

               height_ratio = (y_max-y_min)/height;

               f = figure('visible', 'off');
               hold on;
               subplot(1,3,1)
                imshow(img(roi_y,roi_x,:));
                line(bboxes(j,[1:2:end 1])-double(x_min), bboxes(j,[2:2:end 2])-double(y_min), ... 
                    'Color', [0 1 0], 'LineWidth', 2, 'Marker', 'o');
                line(gt(j,[1:2:end 1])-double(x_min), gt(j,[2:2:end 2])-double(y_min), ... 
                    'Color', [0 0 1], 'LineWidth', 1, 'Marker', 'x');
                title(['Original Image Cut (' num2str(j) ')'])

               txt1 = sprintf('BB: cost(%0.1f), fg-outside(%.01f%%), bg-inside(%.01f%%)\n', ...
                    costs(j), area_scores(j,1)*100, area_scores(j,2)*100);
               txt2 = sprintf('GT: cost(%0.1f), fg-outside(%0.1f%%), bg-inside(%0.1f%%)\n', ...
                    costs_gt(j), area_scores_gt(j,1)*100, area_scores_gt(j,2)*100);

                if (constraints_flags(j) == 1)
                    text(0, height_ratio*1.25*height-40, txt1, 'Color','red');
                else
                    text(0, height_ratio*1.25*height-40, txt1);
                end
                text(0, height_ratio*1.25*height-20, txt2);

               subplot(1,3,2)
                imshow(seg(roi_y,roi_x));
                line(bboxes(j,[1:2:end 1])-double(x_min), bboxes(j,[2:2:end 2])-double(y_min), ... 
                    'Color', [0 1 0], 'LineWidth', 2, 'Marker', 'o');
                line(gt(j,[1:2:end 1])-double(x_min), gt(j,[2:2:end 2])-double(y_min), ... 
                    'Color', [0 0 1], 'LineWidth', 1, 'Marker', 'x');
                title(['Segmentation Mask Cut (' num2str(j) ')']);

               subplot(1,3,3)
                img_masked = uint8(zeros(length(roi_y), length(roi_x), 3));
                img_masked(:,:,1) = img(roi_y,roi_x,1) .* uint8(seg(roi_y,roi_x) > 0);
                img_masked(:,:,2) = img(roi_y,roi_x,2) .* uint8(seg(roi_y,roi_x) > 0);
                img_masked(:,:,3) = img(roi_y,roi_x,3) .* uint8(seg(roi_y,roi_x) > 0);
                imshow(img_masked);
                line(bboxes(j,[1:2:end 1])-double(x_min), bboxes(j,[2:2:end 2])-double(y_min), ... 
                    'Color', [0 1 0], 'LineWidth', 2, 'Marker', 'o');
                line(gt(j,[1:2:end 1])-double(x_min), gt(j,[2:2:end 2])-double(y_min), ... 
                    'Color', [0 0 1], 'LineWidth', 1, 'Marker', 'x');     


               title(['Segmentation Img Cut (' num2str(j) ')']);
               hold off;
               set(f, 'Position', [100, 100, 3*width, height]);
               %remove white borders
               set(gcf, 'PaperUnits', 'points', ... 
                      'PaperPosition', [0 0 3*width height], ...
                      'PaperSize', [width, height], ...
                      'PaperPositionMode','manual',...
                      'InvertHardcopy', 'off',...
                      'Renderer','painters');     %recommended if there are no alphamaps

               file_name = sprintf('%s/%08d.png', seq_dir, j);
               saveas(f, file_name);
               close('all');
           end
        end
    end

end

function [x_min, x_max, y_min, y_max] = get_roi(bbox, size)
    x1 = min(bbox(1:2:end));
    y1 = min(bbox(2:2:end));
    x2 = max(bbox(1:2:end));
    y2 = max(bbox(2:2:end));
    
    w2 = (x2-x1)/2;
    h2 = (y2-y1)/2;
    cx = x1 + w2;
    cy = y1 + h2;

    factor = 1.5;
    
    x_min = max([1 round(cx-factor*w2)]);
    x_max = min([size(1) round(cx+factor*w2)]);
    y_min = max([1 round(cy-factor*h2)]);
    y_max = min([size(2) round(cy+factor*h2)]);
end
