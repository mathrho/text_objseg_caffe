%% Here is the code to generate the bounding box from the heatmap
%
% to reproduce the ILSVRC localization result, you need to first generate
% the heatmap for each testing image by merging the heatmap from the
% 10-crops (it is exactly what the demo code is doing), then resize the merged heatmap back to the original size of
% that image. Then use this bbox generator to generate the bbox from the resized heatmap.
%
% The source code of the bbox generator is also released. Probably you need
% to install the correct version of OpenCV to compile it.
%
% Special thanks to Hui Li for helping on this code.
%
% Bolei Zhou, April 19, 2016

bbox_threshold = [20, 100, 110]; % parameters for the bbox generator
curParaThreshold = [num2str(bbox_threshold(1)) ' ' num2str(bbox_threshold(2)) ' ' num2str(bbox_threshold(3))];

signiture = 'results_lang_seg_sigmoid_thresh0.5';
vf = fopen('/home/zhenyang/Workspace/devel/project/vision/text_obj_track/ILSVRC/test.txt');
% file, trackid, start_frame, end_frame
video_info = textscan(vf,'%s %d %d %d', 'Delimiter', ' ');
fclose(vf);

videos = video_info{1};
video_starts = video_info{2};
video_ends = video_info{3};

counter = 1;
oas_all = {};
query_ids = 0:1;
for vi = 1:numel(videos)
    video = videos{vi}
    for qi = query_ids
        start_frame_id = video_starts(vi);
        end_frame_id = video_ends(vi);

        gtBBoxFile = ['/home/zhenyang/Workspace/data/ImageNetTracker/' video '/groundtruth_rect.txt'];
        gt_bboxes = dlmread(gtBBoxFile);
        num_frame = size(gt_bboxes, 1);
        assert((end_frame_id - start_frame_id + 1) == num_frame);

        fprintf('%d %d\n', qi, num_frame);

        pred_boxes = zeros(num_frame, 4);
        rest_boxes = zeros(num_frame, 4);
        oas = zeros(1, num_frame);
        for fr = start_frame_id:end_frame_id
            im_name = sprintf('%06d', fr);
            curHeatMapFile = ['/home/zhenyang/Workspace/devel/project/vision/text_objseg_caffe/results/ILSVRC/' signiture '/' video '_query_' num2str(qi) '/' im_name '.jpg'];
            curBBoxFile = ['/home/zhenyang/Workspace/devel/project/vision/text_objseg_caffe/results/ILSVRC/' signiture '/' video '_query_' num2str(qi) '/' im_name '.txt'];
            
            frame_counter = fr - start_frame_id + 1;
            % remove last frame ids
            gt_box = gt_bboxes(frame_counter, 1:4);
            gt_box(3) = gt_box(1) + gt_box(3);
            gt_box(4) = gt_box(2) + gt_box(4);

            boxData = dlmread(curBBoxFile);
            boxData_formulate = [boxData(1:4:end)' boxData(2:4:end)' boxData(1:4:end)'+boxData(3:4:end)' boxData(2:4:end)'+boxData(4:4:end)'];
            boxData_formulate = [min(boxData_formulate(:,1),boxData_formulate(:,3)),min(boxData_formulate(:,2),boxData_formulate(:,4)),max(boxData_formulate(:,1),boxData_formulate(:,3)),max(boxData_formulate(:,2),boxData_formulate(:,4))];

            num_box = size(boxData_formulate, 1);
            if num_box > 1
                disp('There are more than 1 boxes generated!')
            end

            max_ov = 0;
            max_sz = 0;
            bebox = boxData_formulate(1, 1:4);
            for bb = 1:num_box
                pred_box = boxData_formulate(bb, 1:4);
                if sum(gt_box) == 0
                    ov = 0;
                else
                    ov = IoU(pred_box, gt_box);
                end
            
                box_sz = (pred_box(3) - pred_box(1))*(pred_box(4) - pred_box(2));
                if box_sz > max_sz
                    max_ov = ov;
                    max_sz = box_sz;
                    bebox = pred_box;
                end
                %max_ov = max(ov, max_ov);
            end

            rest_boxes(frame_counter, :) = bebox;
            pred_boxes(frame_counter, :) = [bebox(1), bebox(2), bebox(3)-bebox(1), bebox(4)-bebox(2)];
            oas(1, frame_counter) = max_ov;
        end

        oas = oas(1, sum(gt_bboxes(:, 1:4), 2)>0);

        predBBoxFile = ['/home/zhenyang/Workspace/devel/project/vision/text_objseg_caffe/results/ILSVRC/' signiture '/' video '_query_' num2str(qi) '_prediction_rect.txt'];
        dlmwrite(predBBoxFile, pred_boxes);

        %restBBoxFile = ['/home/zhenyang/Workspace/devel/project/vision/text_obj_track/ILSVRC/lang_results/results_vgg16_lang_seg_fullconv/' video '_query_' num2str(qi) '_vgg16_lang_seg_fullconv.txt'];
        %dlmwrite(restBBoxFile, rest_boxes);

        oas_all{counter} = oas;
        counter = counter + 1;
    end
end

ovs = cat(2, oas_all{:});
size(ovs)
prec = mean(ovs)
recall = sum(ovs > 0.5) / numel(ovs)

