function s = ScaleScore01(s,scale)
%ScaleScore: scale the scores in user-item rating matrix to [-scale,
%+scale]. See footnote 2.
    s = 2*scale*s-scale;
end