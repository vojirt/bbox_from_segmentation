function [bboxes, costs, scores, constraints_flags, costs_gt, scores_gt] ...
         = optimize_bboxes(segmentation_dir, groundtruth)

    % load image names from the segmentation_dir for most common image
    % extensions
    ext = {'*.jpeg', '*.jpg','*.png'};
    images = [];
    for i = 1:length(ext)
        images = [images dir([segmentation_dir ext{i}])];
    end
    % make the image name with absolute path
    for i = 1:length(images)
        images(i).name = [segmentation_dir images(i).name];
    end
    
    % read groundtruth (for initial solution to optimization)
    gt = dlmread(groundtruth);
    if (size(gt, 1) ~= length(images))
       fprintf('Size of the GT (%d) does not correspond to #images (%d)\n', ...
                size(gt, 1), length(images));
    end
    
    % pre-allocate vars
    size_images = length(images);
    bboxes = nan(size_images, 8);
    constraints_flags = zeros(size_images,1);
    costs = nan(size_images,1);
    scores = nan(size_images, 2);
    costs_gt = nan(size_images,1);
    scores_gt = nan(size_images, 2);

    % parameters of the optimization cost and constraints 
    consts.a = 4;
    consts.b = 1;
    consts.theta_bg = 0.4;
    consts.theta_fg = 0.1;
    consts.theta_fg_simple = 0.05;
    
    % pixels indexes for in/out rectangle computation
    seg_tmp = imread(images(i).name);
    x_img = repmat([(1):(size(seg_tmp,2))], [length([1:size(seg_tmp,1)]) 1]);
    y_img = repmat([(1):(size(seg_tmp,1))]', [1 length([1:size(seg_tmp,2)])]);
    consts.xq = x_img(:, :);
    consts.yq = y_img(:, :);

    
    % process segmention images and optimize bbox for each image 
    for i = 1:size_images
        
        if (mod(i-1,100) == 0)
            id = 100*floor(i/100);
            fprintf('   - Processing frames (%.0f - %.0f)/%.0f\n', ... 
                                            id + 1, min([id+100 size_images]), size_images);
        end
        
        seg = imread(images(i).name);
        
        % use gt if we have it, otherwise used enclosing axis-align bbox 
        % estimated from the segmentation image and min. area rotated bbox
        initial_bbox = [];
        if (i <= size(gt, 1))
            if length(gt(i,:)) == 4
                bb_gt = [gt(i, 1), gt(i, 2), gt(i, 1)+gt(i,3), gt(i, 2), gt(i,1)+gt(i,3), gt(i,2)+gt(i,4), gt(i,1), gt(i,2)+gt(i,4)];
            else
                bb_gt = gt(i,:);
            end
            initial_bbox = [initial_bbox; order_bbox_corners(bb_gt)];
        end
        
        [tmp_y, tmp_x] = find(seg > 0);
        bb_tmp = [min(tmp_x) min(tmp_y) ...
                        max(tmp_x) min(tmp_y) ...
                        max(tmp_x) max(tmp_y) ...
                        min(tmp_x) max(tmp_y)];
        initial_bbox = [initial_bbox; bb_tmp];

        seq_q = seg(:, :);
        X = [consts.xq(seq_q(:) > 0)'; consts.yq(seq_q(:) > 0)'];
        if (size(X,2) > 100)
            initial_bbox = [initial_bbox; order_bbox_corners(minBoundingBox(X))];
        end
        
        % if gt, compute costs for gt bbox
        [cx, cy, width, height, angle] = corners2params(initial_bbox(1,:));
        costs_gt(i) = cost_fnc([cx, cy, width, height, angle], seg, consts);
        scores_gt(i,:) = area_scores(initial_bbox(1,:), seg, consts);   
        
        
        % bind segmentation image and consts to cost functions
        f = @(params)cost_fnc(params, seg, consts);
        g = @(params)constrains_fnc(params, seg, consts);
%       f = @(params)cost_fnc_simple(params, seg, consts);
%       g = @(params)constrains_fnc_simple(params, seg, consts);

        x = [];
        fval = 10^10;
        exitflag_c = -2;

        if (size(X,2) > 100)
            for j = 1:size(initial_bbox,1)
                % initial solution [cx, cy, s_x, s_y, rotation];
                % axis-align rect has angle = 0 in degrees
                [cx, cy, width, height, angle] = corners2params(initial_bbox(j,:));
                params0 = [cx cy width height angle];
                
                % constrained optimization - inequalities
                options = optimoptions('fmincon','Algorithm','interior-point', 'Display', 'off', 'DiffMinChange', 1);
                [x_tmp, fval_tmp, exitflag_c_tmp] = fmincon(f, params0, [], [], [], [], [], [], g, options);
               
                % constraing optimization failed because of constrains
                % -> use unconstrained optimization to get "some" results
                if (exitflag_c_tmp <= -2) 
                    options = optimoptions('fminunc','Algorithm','quasi-newton', 'Display', 'off', 'DiffMinChange', 1);
                    [x_u, fval_u, exitflag_u] = fminunc(f, params0, options);

                    if (fval_u < fval_tmp)
                       x_tmp = x_u;
                       fval_tmp = fval_u;
                       exitflag_c_tmp = exitflag_u;
                    end
                end
                
                % best results so far ?
                if (fval_tmp < fval)
                   x = x_tmp;
                   fval = fval_tmp;
                   exitflag_c = exitflag_c_tmp;
                end
            end
        end
                                         
        % save the bbox and optimization stats
        if (exitflag_c <= -2 && costs_gt(i) <= fval) 
            constraints_flags(i) = 1; 
            bboxes(i,:) = order_bbox_corners(initial_bbox(1,:));
            costs(i) = costs_gt(i);
            scores(i,:) = scores_gt(i,:);
        else
            bboxes(i,:) = params2corners(x);
            costs(i) = fval;
            scores(i,:) = area_scores(bboxes(i,:), seg, consts);
        end
        
    end

end


function [cost] = cost_fnc(params, seg, consts)
    bbox = params2corners(params);
    [~, fg_out_count, bg_in_count] = area_scores(bbox, seg, consts);
    cost = consts.a*fg_out_count + consts.b*bg_in_count;
end

function [cost] = cost_fnc_simple(params, seg, consts)
    bbox = params2corners(params);
    [~, ~, bg_in_count] = area_scores(bbox, seg, consts);
    cost = bg_in_count;
end

function [c, ceq] = constrains_fnc(params, seg, consts)
    bbox = params2corners(params);
    scores = area_scores(bbox, seg, consts);
    c = max([scores(1) - consts.theta_fg, scores(2) - consts.theta_bg]);
    ceq = [];
end

function [c, ceq] = constrains_fnc_simple(params, seg, consts)
    bbox = params2corners(params);
    scores = area_scores(bbox, seg, consts);
    c = scores(1) - consts.theta_fg_simple;
    ceq = [];
end

function [bbox] = params2corners(params)
    angle = pi*params(5)/180;
    R = [cos(angle) -sin(angle); sin(angle) cos(angle)];
    w2 = params(3)/2;
    h2 = params(4)/2;
    cx = params(1);
    cy = params(2);
    
    bbox = [([cx; cy] + R*[-w2; -h2])' ... % x1 - top-left
            ([cx; cy] + R*[w2; -h2])'  ... % x2 - top-right
            ([cx; cy] + R*[w2; h2])'   ... % x3 - bot-right
            ([cx; cy] + R*[-w2; h2])'];    % x4 - bot-left
          
end

function [cx, cy, width, height, angle] = corners2params(bbox)
    cx = 0.25*sum(bbox(1:2:end));
    cy = 0.25*sum(bbox(2:2:end));
    width = sqrt((bbox(1)-bbox(3))^2 + (bbox(2)-bbox(4))^2);
    height = sqrt((bbox(3)-bbox(5))^2 + (bbox(4)-bbox(6))^2);
    angle = atan((bbox(2)-bbox(4))/(bbox(1)-bbox(3)))*180/pi;
end

function [scores, fg_outside_count, bg_inside_count] = area_scores(bbox, seg, consts)
    seq_q = seg(:, :);
    in = inpolygon(consts.xq(:), consts.yq(:), bbox(1:2:end), bbox(2:2:end));

    fg_outside_count = sum(in == 0 & seq_q(:) > 0);
    bg_inside_count = sum(in == 1 & seq_q(:) == 0);
    
    scores = [fg_outside_count/sum(seq_q(:) > 0) bg_inside_count/get_area(bbox)];
end

function [area] = get_area(bbox)
    area = sqrt((bbox(1) - bbox(3))^2 + (bbox(2) - bbox(4))^2) * ...
           sqrt((bbox(3) - bbox(5))^2 + (bbox(4) - bbox(6))^2);
end

function [bbox_order] = order_bbox_corners(bb_tmp)
    bbox_order = zeros(1,8);
    % reorder to [top_left(1,2) top_right(3,4) bot_right(5,6) bot_left(7,8)]
    [~, I] = sort(bb_tmp(2:2:end));
    % top-left and top-right
    if (bb_tmp(2*I(1)-1) < bb_tmp(2*I(2)-1))
        bbox_order(1:2) = bb_tmp((2*I(1)-1):(2*I(1)));
        bbox_order(3:4) = bb_tmp((2*I(2)-1):(2*I(2)));
    else
        bbox_order(3:4) = bb_tmp((2*I(1)-1):(2*I(1)));
        bbox_order(1:2) = bb_tmp((2*I(2)-1):(2*I(2)));        
    end
    % bot-right and bot-left
    if (bb_tmp(2*I(3)-1) > bb_tmp(2*I(4)-1))
        bbox_order(5:6) = bb_tmp((2*I(3)-1):(2*I(3)));
        bbox_order(7:8) = bb_tmp((2*I(4)-1):(2*I(4)));
    else
        bbox_order(7:8) = bb_tmp((2*I(3)-1):(2*I(3)));
        bbox_order(5:6) = bb_tmp((2*I(4)-1):(2*I(4)));
    end
end


% -----------------------------------------------------------------------
function [] = optimize_bboxes_test_func()
    angle = 0.7854;    
    R = [cos(angle) -sin(angle); sin(angle) cos(angle)];    
    x1 = [10;10];
    x2 = [30;10];
    x3 = [30;50];
    x4 = [10;50];
    bbox1 = [x1' x2' x3' x4']

    [cx1, cy1, width1, height1, angle1] = corners2params(bbox1);

    fprintf('BBOX1: center [%.02f %.02f], w h [%.02f %.02f], angle = %.04f/%0.1f\n', ... 
             cx1, cy1, width1, height1, angle1, 180*angle1/pi);
    bbox1_back = params2corners([cx1, cy1, width1, height1, angle1])
       
    c = [cx1;cy1];
    bbox2 = [c'+(R*(x1-c))' c'+(R*(x2-c))' c'+(R*(x3-c))' c'+(R*(x4-c))']
    [cx2, cy2, width2, height2, angle2] = corners2params(bbox2);
    fprintf('BBOX2: center [%.02f %.02f], w h [%.02f %.02f], angle = %.04f/%0.1f\n', ... 
             cx2, cy2, width2, height2, angle2, 180*angle2/pi);
    
    bbox2_back = params2corners([cx2, cy2, width2, height2, angle2])
end

% http://www.mathworks.com/matlabcentral/fileexchange/31126-2d-minimal-bounding-box/content/minBoundingBox.m
function bbox = minBoundingBox(X)
    % compute the minimum bounding box of a set of 2D points
    %   Use:   boundingBox = minBoundingBox(point_matrix)
    %
    % Input:  2xn matrix containing the [x,y] coordinates of n points
    %         *** there must be at least 3 points which are not collinear
    % output: 2x4 matrix containing the coordinates of the bounding box corners
    %
    % Example : generate a random set of point in a randomly rotated rectangle
    %     n = 50000;
    %     t = pi*rand(1);
    %     X = [cos(t) -sin(t) ; sin(t) cos(t)]*[7 0; 0 2]*rand(2,n);
    %     X = [X  20*(rand(2,1)-0.5)];  % add an outlier
    % 
    %     tic
    %     c = minBoundingBox(X);
    %     toc
    % 
    %     figure(42);
    %     hold off,  plot(X(1,:),X(2,:),'.')
    %     hold on,   plot(c(1,[1:end 1]),c(2,[1:end 1]),'r')
    %     axis equal

    % compute the convex hull (CH is a 2*k matrix subset of X)
    k = convhull(X(1,:),X(2,:));
    CH = X(:,k);

    % compute the angle to test, which are the angle of the CH edges as:
    %   "one side of the bounding box contains an edge of the convex hull"
    E = diff(CH,1,2);           % CH edges
    T = atan2(E(2,:),E(1,:));   % angle of CH edges (used for rotation)
    T = unique(mod(T,pi/2));    % reduced to the unique set of first quadrant angles

    % create rotation matrix which contains
    % the 2x2 rotation matrices for *all* angles in T
    % R is a 2n*2 matrix
    R = cos( reshape(repmat(T,2,2),2*length(T),2) ... % duplicate angles in T
           + repmat([0 -pi ; pi 0]/2,length(T),1));   % shift angle to convert sine in cosine

    % rotate CH by all angles
    RCH = R*CH;

    % compute border size  [w1;h1;w2;h2;....;wn;hn]
    % and area of bounding box for all possible edges
    bsize = max(RCH,[],2) - min(RCH,[],2);
    area  = prod(reshape(bsize,2,length(bsize)/2));

    % find minimal area, thus the index of the angle in T 
    [a,i] = min(area);

    % compute the bound (min and max) on the rotated frame
    Rf    = R(2*i+[-1 0],:);   % rotated frame
    bound = Rf * CH;           % project CH on the rotated frame
    bmin  = min(bound,[],2);
    bmax  = max(bound,[],2);

    % compute the corner of the bounding box
    Rf = Rf';
    bb(:,4) = bmax(1)*Rf(:,1) + bmin(2)*Rf(:,2);
    bb(:,1) = bmin(1)*Rf(:,1) + bmin(2)*Rf(:,2);
    bb(:,2) = bmin(1)*Rf(:,1) + bmax(2)*Rf(:,2);
    bb(:,3) = bmax(1)*Rf(:,1) + bmax(2)*Rf(:,2);
    
    bbox = [bb(:,1)' bb(:,2)' bb(:,3)' bb(:,4)'];
    
end






