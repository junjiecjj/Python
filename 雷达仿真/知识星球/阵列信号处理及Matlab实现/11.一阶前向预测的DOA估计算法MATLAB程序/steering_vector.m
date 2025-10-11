function a = steering_vector(phi, wavelength, d, element_num)
% ����������ʸ��
% phi            -- Ŀ�귽λ (0:180)/��
% wavelength     -- ����
% d              -- ��Ԫ���
% element_num    -- ��Ԫ��

n = (0 : element_num - 1).';
a = exp(j * 2 * pi * n * d * cosd(phi) ./ wavelength);